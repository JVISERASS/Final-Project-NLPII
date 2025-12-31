"""
Fine-tuning with Transformers + PEFT (QLoRA)
=============================================
Fine-tunes TinyLlama using QLoRA (Quantized LoRA) with Hugging Face Transformers.

Features:
- 4-bit quantization with NF4
- LoRA for parameter-efficient adaptation
- Gradient checkpointing for memory efficiency
- BFloat16 mixed precision training
- Memory-optimized for consumer GPUs (12GB VRAM)
"""

import os

# Disable tokenizers parallelism warning (must be before importing transformers)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

import torch
import yaml
from datasets import load_from_disk, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
import bitsandbytes as bnb

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


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


def print_trainable_parameters(model):
    """Print the number of trainable parameters in the model."""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    print(f"Model Parameters:")
    print(f"  Trainable: {trainable_params:,} ({100 * trainable_params / all_param:.2f}%)")
    print(f"  Total: {all_param:,}")


class TransformersFinetuner:
    """
    Fine-tuning pipeline using Transformers + PEFT with QLoRA.
    
    Features:
    - 4-bit quantization for memory efficiency
    - LoRA for parameter-efficient fine-tuning
    - Gradient checkpointing for reduced memory
    - BFloat16 mixed precision training
    """
    
    def __init__(
        self,
        config: Optional[dict] = None,
        config_path: str = "configs/config.yaml"
    ):
        self.config = config or load_config(config_path)
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.training_stats = {}
        
        # Set up output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(self.config["output"]["transformers_dir"]) / timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set random seeds
        self._set_seed(self.config["project"]["seed"])
        
    def _set_seed(self, seed: int):
        """Set random seeds for reproducibility."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    def setup_quantization(self) -> BitsAndBytesConfig:
        """Configure 4-bit quantization settings."""
        quant_config = self.config["quantization"]
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=quant_config["load_in_4bit"],
            load_in_8bit=quant_config["load_in_8bit"],
            bnb_4bit_compute_dtype=getattr(torch, quant_config["bnb_4bit_compute_dtype"]),
            bnb_4bit_quant_type=quant_config["bnb_4bit_quant_type"],
            bnb_4bit_use_double_quant=quant_config["bnb_4bit_use_double_quant"]
        )
        
        print("Quantization Configuration:")
        print(f"   4-bit: {quant_config['load_in_4bit']}")
        print(f"   Compute dtype: {quant_config['bnb_4bit_compute_dtype']}")
        print(f"   Quant type: {quant_config['bnb_4bit_quant_type']}")
        print(f"   Double quant: {quant_config['bnb_4bit_use_double_quant']}")
        
        return bnb_config
    
    def setup_lora(self) -> LoraConfig:
        """Configure LoRA settings."""
        lora_config = self.config["lora"]
        
        config = LoraConfig(
            r=lora_config["r"],
            lora_alpha=lora_config["lora_alpha"],
            lora_dropout=lora_config["lora_dropout"],
            target_modules=lora_config["target_modules"],
            bias=lora_config["bias"],
            task_type=TaskType.CAUSAL_LM
        )
        
        print("\nLoRA Configuration:")
        print(f"   Rank (r): {lora_config['r']}")
        print(f"   Alpha: {lora_config['lora_alpha']}")
        print(f"   Dropout: {lora_config['lora_dropout']}")
        print(f"   Target modules: {lora_config['target_modules']}")
        
        return config
    
    def load_model_and_tokenizer(self):
        """Load the base model with quantization and tokenizer."""
        model_name = self.config["model"]["name"]
        print(f"\n Loading model: {model_name}")
        
        # Setup quantization
        bnb_config = self.setup_quantization()
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="right"  # Important for causal LM
        )
        
        # Add special tokens if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        print(f"   Tokenizer loaded. Vocab size: {len(self.tokenizer):,}")
        
        # Load model with quantization
        print(f"   Loading model with 4-bit quantization...")
        print_gpu_memory()
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2" if self._check_flash_attn() else "eager"
        )
        
        print(f"   Model loaded!")
        print_gpu_memory()
        
        # Prepare model for k-bit training
        self.model = prepare_model_for_kbit_training(
            self.model,
            use_gradient_checkpointing=self.config["training"]["gradient_checkpointing"]
        )
        
        # Apply LoRA
        lora_config = self.setup_lora()
        self.model = get_peft_model(self.model, lora_config)
        
        print_trainable_parameters(self.model)
        print_gpu_memory()
        
    def _check_flash_attn(self) -> bool:
        """Check if Flash Attention 2 is available."""
        import flash_attn  # Required dependency
        return True
    
    def load_dataset(self, data_path: str = "data/processed/dolly_processed"):
        """Load the processed dataset."""
        print(f"\n Loading dataset from {data_path}")
        
        self.dataset = load_from_disk(data_path)
        
        print(f"   Train: {len(self.dataset['train']):,} samples")
        print(f"   Validation: {len(self.dataset['validation']):,} samples")
        print(f"   Test: {len(self.dataset['test']):,} samples")
        
        return self.dataset
    
    def tokenize_dataset(self):
        """Tokenize the dataset for training."""
        max_length = self.config["model"]["max_seq_length"]
        print(f"\n🔤 Tokenizing dataset (max_length={max_length})...")
        
        def tokenize_function(examples):
            result = self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=max_length,
                padding=False,
            )
            result["labels"] = result["input_ids"].copy()
            return result
        
        # Use num_proc=None to avoid CUDA multiprocessing issues
        self.tokenized_dataset = self.dataset.map(
            tokenize_function,
            batched=True,
            num_proc=None,  # Single process to avoid CUDA fork issues
            remove_columns=self.dataset["train"].column_names,
            desc="Tokenizing"
        )
        
        print(f"   Tokenization complete!")
        
        return self.tokenized_dataset
    
    def setup_trainer(self):
        """Configure the Trainer with training arguments."""
        train_config = self.config["training"]
        
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            
            # Batch sizes
            per_device_train_batch_size=train_config["per_device_train_batch_size"],
            per_device_eval_batch_size=train_config["per_device_eval_batch_size"],
            gradient_accumulation_steps=train_config["gradient_accumulation_steps"],
            
            # Training duration
            num_train_epochs=train_config["num_train_epochs"],
            max_steps=train_config["max_steps"],
            
            # Learning rate
            learning_rate=train_config["learning_rate"],
            lr_scheduler_type=train_config["lr_scheduler_type"],
            warmup_ratio=train_config["warmup_ratio"],
            weight_decay=train_config["weight_decay"],
            
            # Optimization
            optim=train_config["optim"],
            max_grad_norm=train_config["max_grad_norm"],
            
            # Mixed precision
            bf16=train_config["bf16"],
            fp16=train_config["fp16"],
            
            # Logging & Saving
            logging_steps=train_config["logging_steps"],
            save_steps=train_config["save_steps"],
            eval_steps=train_config["eval_steps"],
            eval_strategy="steps",
            save_total_limit=train_config["save_total_limit"],
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            
            # Efficiency
            gradient_checkpointing=train_config["gradient_checkpointing"],
            dataloader_num_workers=train_config["dataloader_num_workers"],
            dataloader_pin_memory=train_config["dataloader_pin_memory"],
            
            # Logging
            report_to="none",  # Can change to "wandb" if needed
            logging_dir=str(self.output_dir / "logs"),
            
            # Reproducibility
            seed=self.config["project"]["seed"],
            data_seed=self.config["project"]["seed"],
        )
        
        # Data collator for causal LM with proper padding
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            padding=True,
            pad_to_multiple_of=8,  # Efficient for GPU
            return_tensors="pt"
        )
        
        # Initialize Trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_dataset["train"],
            eval_dataset=self.tokenized_dataset["validation"],
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        print("\nTraining Configuration:")
        print(f"   Effective batch size: {train_config['per_device_train_batch_size'] * train_config['gradient_accumulation_steps']}")
        print(f"   Learning rate: {train_config['learning_rate']}")
        print(f"   Epochs: {train_config['num_train_epochs']}")
        print(f"   Optimizer: {train_config['optim']}")
        print(f"   Scheduler: {train_config['lr_scheduler_type']}")
        
        return self.trainer
    
    def train(self):
        """Run the fine-tuning process."""
        print("\n" + "="*60)
        print("Starting Fine-tuning with Transformers + PEFT")
        print("="*60)
        
        # Record start time and memory
        start_time = time.time()
        torch.cuda.reset_peak_memory_stats()
        
        print_gpu_memory()
        
        # Train
        train_result = self.trainer.train()
        
        # Record training stats
        end_time = time.time()
        training_time = end_time - start_time
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3
        
        self.training_stats = {
            "framework": "transformers_peft",
            "training_time_seconds": training_time,
            "training_time_minutes": training_time / 60,
            "peak_memory_gb": peak_memory,
            "train_loss": train_result.training_loss,
            "train_samples": len(self.tokenized_dataset["train"]),
            "epochs": self.config["training"]["num_train_epochs"],
            "effective_batch_size": (
                self.config["training"]["per_device_train_batch_size"] * 
                self.config["training"]["gradient_accumulation_steps"]
            ),
            "lora_rank": self.config["lora"]["r"],
            "learning_rate": self.config["training"]["learning_rate"],
            "timestamp": datetime.now().isoformat()
        }
        
        print("\n" + "="*60)
        print("Training Complete")
        print("="*60)
        print(f"   Training time: {training_time/60:.2f} minutes")
        print(f"   Peak GPU memory: {peak_memory:.2f} GB")
        print(f"   Final train loss: {train_result.training_loss:.4f}")
        
        return train_result
    
    def evaluate(self):
        """Evaluate the fine-tuned model."""
        print("\nRunning evaluation...")
        
        eval_results = self.trainer.evaluate()
        
        self.training_stats["eval_loss"] = eval_results["eval_loss"]
        
        print(f"   Eval loss: {eval_results['eval_loss']:.4f}")
        
        return eval_results
    
    def save_model(self):
        """Save the fine-tuned model and adapter."""
        print("\n💾 Saving model...")
        
        # Save the LoRA adapter
        adapter_path = self.output_dir / "adapter"
        self.model.save_pretrained(str(adapter_path))
        self.tokenizer.save_pretrained(str(adapter_path))
        
        print(f"   Adapter saved to: {adapter_path}")
        
        # Save training stats
        stats_path = self.output_dir / "training_stats.json"
        with open(stats_path, "w") as f:
            json.dump(self.training_stats, f, indent=2)
        
        print(f"   Training stats saved to: {stats_path}")
        
        # Save config
        config_path = self.output_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(self.config, f)
        
        print(f"   Config saved to: {config_path}")
        
        return adapter_path
    
    def merge_and_save(self, output_path: Optional[str] = None):
        """Merge LoRA weights with base model and save."""
        print("\n Merging LoRA weights with base model...")
        
        # Merge weights
        merged_model = self.model.merge_and_unload()
        
        # Save merged model
        if output_path is None:
            output_path = self.output_dir / "merged_model"
        else:
            output_path = Path(output_path)
        
        merged_model.save_pretrained(str(output_path))
        self.tokenizer.save_pretrained(str(output_path))
        
        print(f"   Merged model saved to: {output_path}")
        
        return output_path
    
    def run_full_pipeline(
        self,
        data_path: str = "data/processed/dolly_processed",
        merge_model: bool = True
    ):
        """Run the complete fine-tuning pipeline."""
        print("\n" + "="*70)
        print("Transformers + PEFT Fine-tuning Pipeline")
        print("   Model: " + self.config["model"]["name"])
        print("   GPU: " + self.config["hardware"]["gpu_name"])
        print("="*70 + "\n")
        
        # Load model and tokenizer
        self.load_model_and_tokenizer()
        
        # Load and tokenize dataset
        self.load_dataset(data_path)
        self.tokenize_dataset()
        
        # Setup trainer
        self.setup_trainer()
        
        # Train
        self.train()
        
        # Evaluate
        self.evaluate()
        
        # Save
        self.save_model()
        
        # Optionally merge
        if merge_model:
            self.merge_and_save()
        
        print("\n" + "="*70)
        print(" PIPELINE COMPLETE!")
        print(f"   Output directory: {self.output_dir}")
        print("="*70 + "\n")
        
        return self.training_stats


def main():
    """Main function to run fine-tuning."""
    # Check CUDA
    if not torch.cuda.is_available():
        print(" CUDA not available. This script requires a GPU.")
        sys.exit(1)
    
    print(f" GPU: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA Version: {torch.version.cuda}")
    print(f"   PyTorch Version: {torch.__version__}")
    
    # Initialize and run
    finetuner = TransformersFinetuner(config_path="configs/config.yaml")
    stats = finetuner.run_full_pipeline()
    
    # Print final stats
    print("\nFinal Training Statistics:")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
