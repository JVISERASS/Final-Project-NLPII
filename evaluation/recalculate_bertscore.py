#!/usr/bin/env python3
"""
Re-calculate BERTScore for existing evaluations
================================================
This script recalculates BERTScore for previously evaluated models
using a reliable model (roberta-large) that works well on GPU.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from tqdm import tqdm


def recalculate_bertscore():
    """Recalculate BERTScore for existing evaluations."""
    
    print("=" * 60)
    print(" RE-CALCULATING BERTSCORE")
    print("=" * 60)
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("Warning: Running on CPU - this will be slow")
    
    # Load existing evaluation files
    eval_dir = PROJECT_ROOT / "results" / "evaluation"
    
    eval_files = {
        "transformers": eval_dir / "transformers_eval.json",
        "unsloth": eval_dir / "unsloth_eval.json"
    }
    
    sample_files = {
        "transformers": eval_dir / "transformers_samples.json",
        "unsloth": eval_dir / "unsloth_samples.json"
    }
    
    # Import bert-score
    try:
        from bert_score import score as bert_score
        print("bert-score library loaded successfully")
    except ImportError as e:
        print(f"Error: bert-score not installed: {e}")
        print("Install with: pip install bert-score")
        return
    
    # Process each model
    for model_name, eval_file in eval_files.items():
        if not eval_file.exists():
            print(f"\nSkipping {model_name}: evaluation file not found")
            continue
        
        sample_file = sample_files.get(model_name)
        if not sample_file or not sample_file.exists():
            print(f"\nSkipping {model_name}: samples file not found")
            continue
        
        print(f"\n{'='*60}")
        print(f" Processing: {model_name.upper()}")
        print(f"{'='*60}")
        
        # Load samples
        with open(sample_file, "r") as f:
            samples = json.load(f)
        
        # Extract predictions and references
        predictions = [s["generated"] for s in samples]
        references = [s["reference"] for s in samples]
        
        print(f"Loaded {len(predictions)} samples")
        
        if len(predictions) == 0:
            print("No samples found, skipping...")
            continue
        
        # Calculate BERTScore
        print("Computing BERTScore (using roberta-large)...")
        try:
            P, R, F1 = bert_score(
                predictions,
                references,
                model_type="roberta-large",
                batch_size=8,
                verbose=True,
                device="cuda" if torch.cuda.is_available() else "cpu",
                lang="en"
            )
            
            bertscore_precision = P.mean().item()
            bertscore_recall = R.mean().item()
            bertscore_f1 = F1.mean().item()
            
            print(f"\nResults:")
            print(f"  Precision: {bertscore_precision:.4f}")
            print(f"  Recall:    {bertscore_recall:.4f}")
            print(f"  F1:        {bertscore_f1:.4f}")
            
            # Update evaluation file
            with open(eval_file, "r") as f:
                eval_data = json.load(f)
            
            eval_data["metrics"]["bertscore_precision"] = bertscore_precision
            eval_data["metrics"]["bertscore_recall"] = bertscore_recall
            eval_data["metrics"]["bertscore_f1"] = bertscore_f1
            eval_data["bertscore_model"] = "roberta-large"
            eval_data["bertscore_updated"] = datetime.now().isoformat()
            
            with open(eval_file, "w") as f:
                json.dump(eval_data, f, indent=2)
            
            print(f"Updated: {eval_file}")
            
            # Clean up GPU memory
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error calculating BERTScore: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(" BERTSCORE RECALCULATION COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    recalculate_bertscore()
