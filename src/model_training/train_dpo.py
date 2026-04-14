import os
import torch
import logging
import argparse
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import PeftModel, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig
from datasets import Dataset

from src.model_training.data_prep import prepare_dpo_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(args):
    # We load the base model, but merge it with the SFT adapter for DPO.
    # For DPO, we technically need a reference model and an active model.
    # TRL's DPOTrainer can automatically handle the reference model by deeply cloning the active model before training starts if you don't provide one.
    
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    adapter_dir = "./models/fact_checker_sft"
    logger.info(f"Using base model: {model_id} and SFT adapter from {adapter_dir}")

    # Check hardware — CUDA > MPS (Apple Silicon) > CPU
    has_cuda = torch.cuda.is_available()
    has_mps = torch.backends.mps.is_available()
    if has_cuda:
        device = "cuda"
    elif has_mps:
        device = "mps"
    else:
        device = "cpu"
    logger.info(f"Hardware detection: CUDA={has_cuda}, MPS={has_mps} → using {device.upper()}")

    if not os.path.exists(adapter_dir):
        logger.warning(f"SFT adapter not found at {adapter_dir}. DPO should ideally be run AFTER SFT.")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Setup model loading — 4-bit QLoRA on CUDA, fp16 on MPS, fp32 on CPU
    if has_cuda:
        logger.info("CUDA detected. Loading model in 4-bit...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        model = prepare_model_for_kbit_training(model)
    elif has_mps:
        logger.info("MPS (Apple Silicon) detected. Loading model in fp16...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map={"": device},
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
    else:
        logger.warning("No GPU detected. Loading model in fp32. Training will be slow!")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="cpu",
            torch_dtype=torch.float32,
            trust_remote_code=True
        )

    # Load SFT Adapter if it exists
    if os.path.exists(adapter_dir):
        logger.info("Loading SFT LoRA weights...")
        # Load the PeftModel on top of the base model
        model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=True)
    else:
        # If no SFT, just train the base model directly with Peft
        from peft import LoraConfig, get_peft_model
        peft_config = LoraConfig(
            r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, peft_config)

    # Load Dataset
    logger.info("Loading DPO dataset...")
    # DPOTrainer requires specific columns: prompt, chosen, rejected
    train_dataset = prepare_dpo_dataset("data/processed/train.jsonl", max_samples=args.max_samples)
    val_max = min(args.max_samples // 5, 2000) if args.max_samples else 2000
    val_dataset_raw = prepare_dpo_dataset("data/processed/val.jsonl", max_samples=val_max)

    # We must apply chat template to the datasets for DPO since DPO trainer needs raw strings, not message lists
    def apply_template(examples):
        # examples is a batched dict, we process row by row
        prompts = []
        chosens = []
        rejecteds = []

        for sys_msg, usr_prompt, chosen_msg, rejected_msg in zip(examples['system'], examples['prompt'], examples['chosen'], examples['rejected']):
            # Build the chat for prompt
            prompt_chat = [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": usr_prompt}
            ]
            prompt_str = tokenizer.apply_chat_template(prompt_chat, tokenize=False, add_generation_prompt=True)

            # DPOTrainer expects JUST the assistant response for chosen/rejected, not the whole conversation
            chosen_str = chosen_msg[0]['content'] + tokenizer.eos_token
            rejected_str = rejected_msg[0]['content'] + tokenizer.eos_token

            prompts.append(prompt_str)
            chosens.append(chosen_str)
            rejecteds.append(rejected_str)

        return {"prompt": prompts, "chosen": chosens, "rejected": rejecteds}

    # Remove original columns to keep only the 3 required by DPO
    train_dataset = train_dataset.map(
        apply_template,
        batched=True,
        remove_columns=train_dataset.column_names
    )
    val_dataset = val_dataset_raw.map(
        apply_template,
        batched=True,
        remove_columns=val_dataset_raw.column_names
    )

    # Training Arguments
    output_dir = "./models/fact_checker_dpo"
    training_args = DPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=1e-5, # DPO learning rate is usually lower than SFT
        logging_steps=10,
        max_steps=args.max_steps if args.max_steps else -1,
        num_train_epochs=args.epochs if not args.max_steps else 1,
        eval_strategy="epoch",
        remove_unused_columns=False, # Required for DPO Trainer
        optim="paged_adamw_8bit", # 8-bit math frees VRAM allowing faster throughput mapping
        # dataloader workers: GPU pipelining only helps when CUDA is available.
        dataloader_num_workers=2 if has_cuda else 0,
        fp16=False, # Disable GradScaler to bypass TinyLlama bfloat16 poisoning
        bf16=False,
        use_cpu=device == "cpu",
        report_to="none",
        beta=0.4 # Increased KL penalty to more aggressively punish hallucinations
    )

    # DPO Trainer
    logger.info("Initializing DPO Trainer...")

    trainer = DPOTrainer(
        model=model,
        ref_model=None, # TRL uses PEFT adapter disable/enable to derive the reference model implicitly
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        args=training_args
    )
    
    logger.info("Starting DPO training...")
    trainer.train()
    
    logger.info("DPO Training complete. Saving final DPO adapter...")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Model saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_samples", type=int, default=2000, help="Limit number of dataset samples for debugging")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--max_steps", type=int, default=None, help="Override epochs with max steps")
    args = parser.parse_args()
    
    main(args)
