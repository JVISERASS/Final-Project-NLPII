#!/usr/bin/env python3
"""
Calculate all metrics for base model including BERTScore
========================================================
This script calculates BLEU, ROUGE, and BERTScore for the base model
using the existing base_model_samples.json file.
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


def calculate_base_model_metrics():
    """Calculate all metrics for base model."""
    
    print("=" * 60)
    print(" CALCULATING BASE MODEL METRICS")
    print("=" * 60)
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load base model samples
    eval_dir = PROJECT_ROOT / "results" / "evaluation"
    sample_file = eval_dir / "base_model_samples.json"
    
    if not sample_file.exists():
        print(f"Error: {sample_file} not found!")
        return
    
    with open(sample_file, "r") as f:
        samples = json.load(f)
    
    # Extract predictions and references
    predictions = [s["generated"] for s in samples]
    references = [s["reference"] for s in samples]
    
    print(f"Loaded {len(predictions)} samples")
    
    if len(predictions) == 0:
        print("No samples found!")
        return
    
    # Calculate BLEU scores
    print("\nCalculating BLEU scores...")
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    
    smoothie = SmoothingFunction().method1
    bleu_scores = []
    bleu1_scores = []
    bleu2_scores = []
    
    for pred, ref in zip(predictions, references):
        pred_tokens = pred.lower().split()
        ref_tokens = [ref.lower().split()]
        
        bleu = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smoothie)
        bleu1 = sentence_bleu(ref_tokens, pred_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothie)
        bleu2 = sentence_bleu(ref_tokens, pred_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)
        
        bleu_scores.append(bleu * 100)
        bleu1_scores.append(bleu1 * 100)
        bleu2_scores.append(bleu2 * 100)
    
    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    avg_bleu1 = sum(bleu1_scores) / len(bleu1_scores)
    avg_bleu2 = sum(bleu2_scores) / len(bleu2_scores)
    
    print(f"  BLEU: {avg_bleu:.2f}")
    print(f"  BLEU-1: {avg_bleu1:.2f}")
    print(f"  BLEU-2: {avg_bleu2:.2f}")
    
    # Calculate ROUGE scores
    print("\nCalculating ROUGE scores...")
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)
    
    avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores)
    avg_rouge2 = sum(rouge2_scores) / len(rouge2_scores)
    avg_rougeL = sum(rougeL_scores) / len(rougeL_scores)
    
    print(f"  ROUGE-1: {avg_rouge1:.4f}")
    print(f"  ROUGE-2: {avg_rouge2:.4f}")
    print(f"  ROUGE-L: {avg_rougeL:.4f}")
    
    # Calculate BERTScore
    print("\nCalculating BERTScore (using roberta-large)...")
    try:
        from bert_score import score as bert_score
        
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
        
        print(f"  BERTScore Precision: {bertscore_precision:.4f}")
        print(f"  BERTScore Recall: {bertscore_recall:.4f}")
        print(f"  BERTScore F1: {bertscore_f1:.4f}")
        
    except Exception as e:
        print(f"Error calculating BERTScore: {e}")
        bertscore_precision = 0.0
        bertscore_recall = 0.0
        bertscore_f1 = 0.0
    
    # Compile all metrics
    metrics = {
        "bleu": avg_bleu,
        "bleu_1": avg_bleu1,
        "bleu_2": avg_bleu2,
        "rouge_1": avg_rouge1,
        "rouge_2": avg_rouge2,
        "rouge_l": avg_rougeL,
        "bertscore_precision": bertscore_precision,
        "bertscore_recall": bertscore_recall,
        "bertscore_f1": bertscore_f1
    }
    
    # Save results
    output_file = eval_dir / "base_model_eval.json"
    results = {
        "model": "TinyLlama-1.1B-Chat (Base)",
        "num_samples": len(predictions),
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics
    }
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f" RESULTS SAVED TO: {output_file}")
    print(f"{'='*60}")
    
    # Also update before_after_comparison.json
    comparison_file = eval_dir / "before_after_comparison.json"
    if comparison_file.exists():
        with open(comparison_file, "r") as f:
            comparison = json.load(f)
        
        comparison["base_model"] = {
            "bleu": avg_bleu,
            "bleu_1": avg_bleu1,
            "bleu_2": avg_bleu2,
            "rouge_1": avg_rouge1,
            "rouge_2": avg_rouge2,
            "rouge_l": avg_rougeL,
            "bertscore_precision": bertscore_precision,
            "bertscore_recall": bertscore_recall,
            "bertscore_f1": bertscore_f1
        }
        comparison["timestamp"] = datetime.now().isoformat()
        
        with open(comparison_file, "w") as f:
            json.dump(comparison, f, indent=2)
        
        print(f"Updated: {comparison_file}")
    
    return metrics


if __name__ == "__main__":
    calculate_base_model_metrics()
