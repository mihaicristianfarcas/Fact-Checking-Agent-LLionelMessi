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
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    logger.info(f"Using base model: {model_id}")

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

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Causal LM: pad from left so eos isn't mid-sequence

    # Setup model loading — 4-bit QLoRA on CUDA, fp16 on MPS, fp32 on CPU
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
            torch_dtype=torch.float16,
            attn_implementation="sdpa" # Native PyTorch flash-attention alternative
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

    # Setup LoRA
    logger.info("Setting up LoRA adapters...")
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        # Cover all attention projections + MLP layers for fuller adaptation.
        # q/k/v/o = attention; gate/up/down = MLP (LLaMA-style SwiGLU FFN).
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    # Load Dataset
    logger.info("Loading SFT dataset...")
    train_dataset = prepare_sft_dataset("data/processed/train.jsonl", tokenizer=tokenizer, max_samples=args.max_samples)
    # Val is 20% of train samples, capped at 2000 to avoid wasting eval time.
    val_max = min(args.max_samples // 5, 2000) if args.max_samples else 2000
    val_dataset = prepare_sft_dataset("data/processed/val.jsonl", tokenizer=tokenizer, max_samples=val_max)

    # Training Arguments
    output_dir = "./models/fact_checker_sft"
    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        warmup_steps=100,
        max_steps=args.max_steps if args.max_steps else -1,
        num_train_epochs=args.epochs if not args.max_steps else 1,
        eval_strategy="epoch",  # Evaluates once exactly at the end
        save_strategy="epoch",  # Saves backup once exactly at the end
        # paged_adamw_8bit is CUDA-only; fall back to standard AdamW on MPS/CPU.
        optim="paged_adamw_8bit" if has_cuda else "adamw_torch",
        # dataloader workers: GPU pipelining only helps when CUDA is available.
        dataloader_num_workers=2 if has_cuda else 0,
        # pin_memory is not supported on MPS and triggers a warning.
        dataloader_pin_memory=has_cuda,
        fp16=has_cuda,  # Safe with QLoRA; bf16 is the problematic one (TinyLlama config.json)
        bf16=False,
        use_cpu=device == "cpu",
        report_to="none", # Turn off wandb for local debug
        # Explicit context length — TinyLlama supports 2048; evidence prompts can be long.
        max_length=2048,
    )

    # SFT Trainer
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=training_args
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
