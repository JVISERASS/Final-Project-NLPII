#!/usr/bin/env python3
"""
NLP2 Final Project - Unsloth Fine-tuning (Simplified)
======================================================
Fine-tunes TinyLlama using Unsloth's optimized pipeline.
"""

# CRITICAL: Import unsloth FIRST before anything else
import unsloth

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

import torch
import yaml
from datasets import load_from_disk

# Now import the rest
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer, SFTConfig


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def print_gpu_memory():
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")


class UnslothFinetuner:
    """Unsloth fine-tuning wrapper class for pipeline integration."""
    
    def __init__(self, config: dict = None, config_path: str = "configs/config.yaml"):
        self.config = config or load_config(config_path)
        self.output_dir = None
        self.model = None
        self.tokenizer = None
        
    def run_full_pipeline(self, data_path: str = "data/processed/dolly_processed", merge_model: bool = True) -> dict:
        """Run the complete Unsloth fine-tuning pipeline."""
        print("="*70)
        print("UNSLOTH FINE-TUNING PIPELINE")
        print("="*70)
        
        config = self.config
        
        # Setup output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(config["output"]["unsloth_dir"]) / timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set seed
        torch.manual_seed(config["project"]["seed"])
        torch.cuda.manual_seed_all(config["project"]["seed"])
        
        print(f"\nLoading model: {config['model']['name']}")
        print_gpu_memory()
        
        # Load model with Unsloth - 4-bit quantization
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config["model"]["name"],
            max_seq_length=config["model"]["max_seq_length"],
            dtype=None,
            load_in_4bit=True,
        )
        
        print("Model loaded!")
        print_gpu_memory()
        
        # Add LoRA adapters
        print("\nAdding LoRA adapters...")
        lora_config = config["lora"]
        
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_config["r"],
            lora_alpha=lora_config["lora_alpha"],
            lora_dropout=lora_config["lora_dropout"],
            target_modules=lora_config["target_modules"],
            bias=lora_config["bias"],
            use_gradient_checkpointing="unsloth",
            random_state=config["project"]["seed"],
            use_rslora=True,
        )
        
        # Print trainable parameters
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
        
        # Setup tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # Load dataset
        print("\nLoading dataset...")
        dataset = load_from_disk(data_path)
        print(f"   Train: {len(dataset['train']):,}")
        print(f"   Validation: {len(dataset['validation']):,}")
        
        # Training config
        train_config = config["training"]
        
        sft_config = SFTConfig(
            output_dir=str(self.output_dir),
            per_device_train_batch_size=train_config["per_device_train_batch_size"],
            per_device_eval_batch_size=train_config["per_device_eval_batch_size"],
            gradient_accumulation_steps=train_config["gradient_accumulation_steps"],
            num_train_epochs=train_config["num_train_epochs"],
            max_steps=train_config["max_steps"],
            learning_rate=train_config["learning_rate"],
            lr_scheduler_type=train_config["lr_scheduler_type"],
            warmup_ratio=train_config["warmup_ratio"],
            weight_decay=train_config["weight_decay"],
            optim="adamw_8bit",
            max_grad_norm=train_config["max_grad_norm"],
            bf16=is_bfloat16_supported(),
            fp16=not is_bfloat16_supported(),
            logging_steps=train_config["logging_steps"],
            save_steps=train_config["save_steps"],
            eval_steps=train_config["eval_steps"],
            eval_strategy="steps",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="none",
            seed=config["project"]["seed"],
            dataset_text_field="text",
            max_length=config["model"]["max_seq_length"],
            packing=False,
            dataset_num_proc=None,
        )
        
        # Create trainer
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            args=sft_config,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
        )
        
        print("\n" + "="*70)
        print("Starting Training")
        print("="*70)
        
        # Record start time
        start_time = time.time()
        torch.cuda.reset_peak_memory_stats()
        
        # Train
        train_result = trainer.train()
        
        # Record stats
        training_time = time.time() - start_time
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3
        
        print("\n" + "="*70)
        print("Training Complete")
        print("="*70)
        print(f"   Time: {training_time/60:.2f} minutes")
        print(f"   Peak memory: {peak_memory:.2f} GB")
        print(f"   Train loss: {train_result.training_loss:.4f}")
        
        # Evaluate
        print("\nRunning evaluation...")
        eval_results = trainer.evaluate()
        print(f"   Eval loss: {eval_results['eval_loss']:.4f}")
        
        # Save training stats
        stats = {
            "framework": "unsloth",
            "training_time_seconds": training_time,
            "training_time_minutes": training_time / 60,
            "peak_memory_gb": peak_memory,
            "train_loss": train_result.training_loss,
            "eval_loss": eval_results["eval_loss"],
            "train_samples": len(dataset["train"]),
            "epochs": train_config["num_train_epochs"],
            "effective_batch_size": train_config["per_device_train_batch_size"] * train_config["gradient_accumulation_steps"],
            "lora_rank": lora_config["r"],
            "learning_rate": train_config["learning_rate"],
            "timestamp": datetime.now().isoformat()
        }
        
        with open(self.output_dir / "training_stats.json", "w") as f:
            json.dump(stats, f, indent=2)
        
        # Save adapter
        print("\nSaving adapter...")
        adapter_dir = self.output_dir / "adapter"
        model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
        
        # Save merged model
        if merge_model:
            print("Saving merged model...")
            merged_dir = self.output_dir / "merged_model"
            model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")
        
        # Save config
        with open(self.output_dir / "config.yaml", "w") as f:
            yaml.dump(config, f)
        
        print("\n" + "="*70)
        print("UNSLOTH TRAINING COMPLETE!")
        print("="*70)
        print(f"\nOutput directory: {self.output_dir}")
        
        self.model = model
        self.tokenizer = tokenizer
        
        return stats


def main():
    """Main entry point for standalone execution."""
    finetuner = UnslothFinetuner()
    finetuner.run_full_pipeline()


if __name__ == "__main__":
    main()
