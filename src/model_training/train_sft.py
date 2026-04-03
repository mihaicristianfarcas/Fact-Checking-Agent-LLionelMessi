import torch
import logging
import argparse
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

from src.model_training.data_prep import prepare_sft_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(args):
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    logger.info(f"Using base model: {model_id}")

    # Check hardware
    has_cuda = torch.cuda.is_available()
    device = "cuda" if has_cuda else "cpu"
    logger.info(f"Hardware detection: CUDA Available = {has_cuda}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Setup Quantization if CUDA is available, otherwise load normally
    if has_cuda:
        logger.info("CUDA detected. Loading model in 4-bit using QLoRA...")
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
    else:
        logger.warning("No CUDA detected. Loading model in standard precision. Training will be slow!")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="cpu", # Fallback to CPU for Intel/Integrated Graphics without ROCm/CUDA
            torch_dtype=torch.float32,
            trust_remote_code=True
        )

    # Setup LoRA
    logger.info("Setting up LoRA adapters...")
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"] # Basic targeting for attention layers
    )

    # Load Dataset
    logger.info("Loading SFT dataset...")
    train_dataset = prepare_sft_dataset("data/processed/train.jsonl", tokenizer=tokenizer, max_samples=args.max_samples)
    val_dataset = prepare_sft_dataset("data/processed/val.jsonl", tokenizer=tokenizer, max_samples=args.max_samples // 5 if args.max_samples else None)

    # Training Arguments
    output_dir = "./models/fact_checker_sft"
    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        max_steps=args.max_steps if args.max_steps else -1,
        num_train_epochs=args.epochs if not args.max_steps else 1,
        eval_strategy="steps",
        eval_steps=400,
        save_strategy="steps",
        save_steps=400,
        optim="paged_adamw_8bit", # 8-bit math frees VRAM allowing faster throughput mapping
        dataloader_num_workers=2, # Streams data from CPU to GPU in parallel
        max_seq_length=1024,      # Caps extreme padding from slowing down attention matrix
        fp16=False,
        bf16=False,
        use_cpu=not has_cuda,
        report_to="none", # Turn off wandb for local debug

    )

    # SFT Trainer
    
    # ---------------------------
    # CRITICAL COLAB T4 FIX: 
    # Qwen natively forces some weights (like PEFT parameters or unquantized heads) 
    # to bfloat16. T4 GPUs cannot do math on bfloat16. We MUST downcast them securely.
    for name, param in model.named_parameters():
        if param.dtype == torch.bfloat16:
            param.data = param.data.to(torch.float16)
    for name, buffer in model.named_buffers():
        if buffer.dtype == torch.bfloat16:
            buffer.data = buffer.data.to(torch.float16)
    if hasattr(model, "config") and hasattr(model.config, "torch_dtype"):
        if model.config.torch_dtype == torch.bfloat16:
            model.config.torch_dtype = torch.float16
    # ---------------------------
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=training_args,
    )

    # Note: TRL uses the model's chat template automatically if "messages" key exists.
    logger.info("Starting SFT training...")
    trainer.train()
    
    logger.info("Training complete. Saving adapter model...")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Model saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_samples", type=int, default=1000, help="Limit number of dataset samples for debugging")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--max_steps", type=int, default=None, help="Override epochs with max steps")
    args = parser.parse_args()
    
    main(args)
