"""
NLP2 Final Project - Data Preparation Module
============================================
Prepares the Databricks Dolly 15k dataset for fine-tuning.

Features:
- Automatic download and caching
- Train/Val/Test split (12k/2k/1k)
- Instruction formatting for chat models
- Support for multiple prompt templates
"""

import os
import json
import random
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import torch
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer
import yaml


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def format_instruction_tinyllama(sample: Dict) -> str:
    """
    Format instruction-response pair for TinyLlama chat template.
    
    TinyLlama uses the ChatML format:
    <|system|>
    {system_message}</s>
    <|user|>
    {user_message}</s>
    <|assistant|>
    {assistant_message}</s>
    """
    instruction = sample.get("instruction", "")
    context = sample.get("context", "")
    response = sample.get("response", "")
    
    # Build the user message with context if available
    if context and context.strip():
        user_message = f"{instruction}\n\nContext: {context}"
    else:
        user_message = instruction
    
    # TinyLlama ChatML format
    formatted = f"""<|system|>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible.</s>
<|user|>
{user_message}</s>
<|assistant|>
{response}</s>"""
    
    return formatted


def format_for_evaluation(sample: Dict) -> Dict:
    """
    Format sample for evaluation - separate prompt and reference.
    Returns dict with 'prompt' and 'reference' keys.
    """
    instruction = sample.get("instruction", "")
    context = sample.get("context", "")
    response = sample.get("response", "")
    category = sample.get("category", "")
    
    if context and context.strip():
        user_message = f"{instruction}\n\nContext: {context}"
    else:
        user_message = instruction
    # Dolly es un dataset de conocimiento general por eso en el system no se menciona nada de un dominio especifico
    prompt = f"""<|system|>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible.</s> 
<|user|>
{user_message}</s>
<|assistant|>
"""
    
    return {
        "prompt": prompt,
        "reference": response,
        "instruction": instruction,
        "context": context,
        "category": category
    }


class DollyDataProcessor:
    """
    Processes the Databricks Dolly 15k dataset for LLM fine-tuning.
    
    Attributes:
        config: Configuration dictionary
        tokenizer: HuggingFace tokenizer
        format_func: Function to format instruction-response pairs
    """
    
    def __init__(
        self,
        config: Optional[dict] = None,
        config_path: str = "configs/config.yaml",
        tokenizer: Optional[AutoTokenizer] = None
    ):
        self.config = config or load_config(config_path)
        self.tokenizer = tokenizer
        
        # Use TinyLlama ChatML format
        self.format_func = format_instruction_tinyllama
        
        # Set random seed for reproducibility
        self.seed = self.config["project"]["seed"]
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        
    def load_raw_dataset(self) -> Dataset:
        """Load the raw Dolly dataset from HuggingFace."""
        print("Loading Databricks Dolly 15k dataset...")
        dataset = load_dataset(
            self.config["dataset"]["name"],
            split="train"  # Dolly only has train split
        )
        print(f"Loaded {len(dataset)} samples")
        return dataset
    
    def create_splits(self, dataset: Dataset) -> DatasetDict:
        """
        Split dataset into train/validation/test sets.
        
        Default split: 12000/2000/1000 (or proportionally if max_samples set)
        """
        train_size = self.config["dataset"]["train_size"]
        val_size = self.config["dataset"]["val_size"]
        test_size = self.config["dataset"]["test_size"]
        max_samples = self.config["dataset"].get("max_samples")
        
        # Handle downsampling for debugging
        if max_samples is not None:
            total = train_size + val_size + test_size
            ratio = max_samples / total
            train_size = int(train_size * ratio)
            val_size = int(val_size * ratio)
            test_size = max_samples - train_size - val_size
            print(f"WARNING: Downsampling to {max_samples} samples")
        
        # Shuffle and split
        dataset = dataset.shuffle(seed=self.seed)
        
        total_needed = train_size + val_size + test_size
        if len(dataset) < total_needed:
            print(f"WARNING: Dataset has only {len(dataset)} samples, adjusting splits...")
            ratio = len(dataset) / total_needed
            train_size = int(train_size * ratio)
            val_size = int(val_size * ratio)
            test_size = len(dataset) - train_size - val_size
        
        train_dataset = dataset.select(range(train_size))
        val_dataset = dataset.select(range(train_size, train_size + val_size))
        test_dataset = dataset.select(range(train_size + val_size, train_size + val_size + test_size))
        
        splits = DatasetDict({
            "train": train_dataset,
            "validation": val_dataset,
            "test": test_dataset
        })
        
        print("Dataset splits created:")
        print(f"  Train:      {len(splits['train']):,} samples")
        print(f"  Validation: {len(splits['validation']):,} samples")
        print(f"  Test:       {len(splits['test']):,} samples")
        
        return splits
    
    def format_dataset(self, dataset: DatasetDict) -> DatasetDict:
        """Apply instruction formatting to all splits."""
        print("Formatting instructions...")
        
        def add_formatted_text(sample):
            sample["text"] = self.format_func(sample)
            return sample
        
        formatted = dataset.map(
            add_formatted_text,
            num_proc=4,
            desc="Formatting"
        )
        
        return formatted
    
    def prepare_for_evaluation(self, dataset: Dataset) -> Dataset:
        """Prepare test set for evaluation with prompts and references."""
        print("Preparing evaluation dataset...")
        
        eval_dataset = dataset.map(
            lambda x: format_for_evaluation(x),
            num_proc=4,
            desc="Preparing eval"
        )
        
        return eval_dataset
    
    def tokenize_dataset(
        self,
        dataset: DatasetDict,
        max_length: Optional[int] = None
    ) -> DatasetDict:
        """
        Tokenize the formatted dataset.
        
        Args:
            dataset: Formatted DatasetDict
            max_length: Maximum sequence length (default from config)
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not provided. Initialize with tokenizer.")
        
        max_length = max_length or self.config["model"]["max_seq_length"]
        print(f"Tokenizing with max_length={max_length}...")
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=max_length,
                padding=False,  # Dynamic padding during training
                return_tensors=None
            )
        
        tokenized = dataset.map(
            tokenize_function,
            batched=True,
            num_proc=4,
            remove_columns=dataset["train"].column_names,
            desc="Tokenizing"
        )
        
        return tokenized
    
    def get_dataset_statistics(self, dataset: DatasetDict) -> Dict:
        """Calculate and return dataset statistics."""
        stats = {}
        
        for split_name, split_data in dataset.items():
            # Category distribution
            categories = {}
            text_lengths = []
            
            for sample in split_data:
                cat = sample.get("category", "unknown")
                categories[cat] = categories.get(cat, 0) + 1
                
                if "text" in sample:
                    text_lengths.append(len(sample["text"]))
            
            stats[split_name] = {
                "num_samples": len(split_data),
                "categories": categories,
                "avg_text_length": sum(text_lengths) / len(text_lengths) if text_lengths else 0,
                "max_text_length": max(text_lengths) if text_lengths else 0,
                "min_text_length": min(text_lengths) if text_lengths else 0
            }
        
        return stats
    
    def save_dataset(self, dataset: DatasetDict, output_dir: str = "data/processed"):
        """Save processed dataset to disk."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        dataset.save_to_disk(str(output_path / "dolly_processed"))
        print(f"Dataset saved to {output_path / 'dolly_processed'}")
        
        # Also save as JSON for inspection
        for split_name, split_data in dataset.items():
            json_path = output_path / f"{split_name}.json"
            split_data.to_json(str(json_path))
            print(f"{split_name}.json saved")
    
    def prepare_all(
        self,
        save: bool = True,
        output_dir: str = "data/processed"
    ) -> Tuple[DatasetDict, DatasetDict]:
        """
        Full pipeline: load, split, format, and optionally save.
        
        Returns:
            Tuple of (formatted_dataset, eval_dataset)
        """
        print("\n" + "="*60)
        print("DOLLY DATASET PREPARATION PIPELINE")
        print("="*60 + "\n")
        
        # Load and split
        raw_dataset = self.load_raw_dataset()
        splits = self.create_splits(raw_dataset)
        
        # Format for training
        formatted = self.format_dataset(splits)
        
        # Prepare evaluation set
        eval_dataset = self.prepare_for_evaluation(splits["test"])
        
        # Get statistics
        stats = self.get_dataset_statistics(formatted)
        print("\nDataset Statistics:")
        for split_name, split_stats in stats.items():
            print(f"\n   {split_name.upper()}:")
            print(f"Samples: {split_stats['num_samples']:,}")
            print(f"Avg text length: {split_stats['avg_text_length']:.0f} chars")
            print(f"Categories: {len(split_stats['categories'])}")
        
        # Save if requested
        if save:
            self.save_dataset(formatted, output_dir)
            
            # Save evaluation dataset separately
            eval_path = Path(output_dir) / "eval_dataset"
            eval_path.mkdir(parents=True, exist_ok=True)
            eval_dataset.to_json(str(eval_path / "test_eval.json"))
            print(f"Evaluation dataset saved to {eval_path}")
        
        print("\n" + "="*60)
        print("DATA PREPARATION COMPLETE")
        print("="*60 + "\n")
        
        return formatted, eval_dataset


def main():
    """Main function to prepare the dataset."""
    # Initialize processor
    processor = DollyDataProcessor(
        config_path="configs/config.yaml"
    )
    
    # Run full preparation pipeline
    formatted_dataset, eval_dataset = processor.prepare_all(
        save=True,
        output_dir="data/processed"
    )
    
    # Print some examples
    print("\nSample formatted instruction:\n")
    print("-" * 50)
    print(formatted_dataset["train"][0]["text"][:500] + "...")
    print("-" * 50)


if __name__ == "__main__":
    main()
