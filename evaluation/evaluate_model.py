#!/usr/bin/env python3
"""
Model Evaluation Script
=======================
Evaluates fine-tuned models using BLEU, ROUGE, and BERTScore.
Supports both Transformers and Unsloth models.
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from datasets import load_dataset
from tqdm import tqdm


def load_test_data(num_samples: int = 100):
    """Load test samples from Dolly dataset."""
    print(f" Loading test data ({num_samples} samples)...")
    
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
    
    # Use last 1000 samples for test set
    test_indices = list(range(len(dataset) - 1000, len(dataset)))
    test_data = dataset.select(test_indices[:num_samples])
    
    prompts = []
    references = []
    
    for sample in test_data:
        instruction = sample["instruction"]
        context = sample.get("context", "")
        response = sample["response"]
        
        if context:
            prompt = f"### Instruction:\n{instruction}\n\n### Context:\n{context}\n\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
        
        prompts.append(prompt)
        references.append(response)
    
    return prompts, references


def generate_with_transformers(model_path: str, prompts: list, max_new_tokens: int = 128):
    """Generate responses using Transformers."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f" Loading Transformers model: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    generations = []
    print(f"Generating {len(prompts)} responses...")
    
    for prompt in tqdm(prompts, desc="Generating"):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove prompt from output
        response = generated_text[len(prompt):].strip()
        generations.append(response)
    
    # Cleanup
    del model
    torch.cuda.empty_cache()
    
    return generations


def evaluate_model(
    model_path: str,
    model_name: str,
    num_samples: int = 100,
    include_bertscore: bool = True,
    output_dir: str = "results/evaluation"
):
    """Evaluate a model with BLEU, ROUGE, and BERTScore."""
    
    print("=" * 60)
    print(f" EVALUATING MODEL: {model_name}")
    print("=" * 60)
    print(f"Model path: {model_path}")
    print(f"Samples: {num_samples}")
    print(f"BERTScore: {include_bertscore}")
    print("=" * 60)
    
    # Load test data
    prompts, references = load_test_data(num_samples)
    
    # Generate responses
    print("\n Generating responses...")
    start_time = time.time()
    predictions = generate_with_transformers(model_path, prompts)
    generation_time = time.time() - start_time
    print(f"Generation time: {generation_time:.2f}s")
    
    # Calculate metrics
    print("\n Calculating metrics...")
    
    # BLEU
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.tokenize import word_tokenize
    import nltk
    
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)
    
    smoothing = SmoothingFunction()
    bleu_scores = []
    bleu_1_scores = []
    bleu_2_scores = []
    
    print(f"Computing BLEU...")
    for pred, ref in zip(predictions, references):
        pred_tokens = word_tokenize(pred.lower())
        ref_tokens = [word_tokenize(ref.lower())]
        
        try:
            bleu = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smoothing.method1)
            bleu_1 = sentence_bleu(ref_tokens, pred_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing.method1)
            bleu_2 = sentence_bleu(ref_tokens, pred_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing.method1)
        except:
            bleu, bleu_1, bleu_2 = 0.0, 0.0, 0.0
        
        bleu_scores.append(bleu)
        bleu_1_scores.append(bleu_1)
        bleu_2_scores.append(bleu_2)
    
    avg_bleu = sum(bleu_scores) / len(bleu_scores) * 100
    avg_bleu_1 = sum(bleu_1_scores) / len(bleu_1_scores) * 100
    avg_bleu_2 = sum(bleu_2_scores) / len(bleu_2_scores) * 100
    
    # ROUGE
    from rouge_score import rouge_scorer
    
    print(f"Computing ROUGE...")
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    rouge_1_scores = []
    rouge_2_scores = []
    rouge_l_scores = []
    
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        rouge_1_scores.append(scores['rouge1'].fmeasure)
        rouge_2_scores.append(scores['rouge2'].fmeasure)
        rouge_l_scores.append(scores['rougeL'].fmeasure)
    
    avg_rouge_1 = sum(rouge_1_scores) / len(rouge_1_scores)
    avg_rouge_2 = sum(rouge_2_scores) / len(rouge_2_scores)
    avg_rouge_l = sum(rouge_l_scores) / len(rouge_l_scores)
    
    # BERTScore
    bertscore_f1 = 0.0
    bertscore_precision = 0.0
    bertscore_recall = 0.0

    if include_bertscore:
        print(f"Computing BERTScore...")
        try:
            from bert_score import score as bert_score
            
            # Use a smaller, more reliable model for BERTScore
            # roberta-large is the default and most stable
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
            print(f"  BERTScore P: {bertscore_precision:.4f}, R: {bertscore_recall:.4f}, F1: {bertscore_f1:.4f}")
        except ImportError as e:
            print(f"Warning: bert-score not installed: {e}")
            print("  Install with: pip install bert-score")
        except Exception as e:
            print(f"Warning: BERTScore calculation failed: {e}")
            import traceback
            traceback.print_exc()    # Results
    results = {
        "model_name": model_name,
        "model_path": model_path,
        "timestamp": datetime.now().isoformat(),
        "num_samples": num_samples,
        "generation_time_s": generation_time,
        "metrics": {
            "bleu": avg_bleu,
            "bleu_1": avg_bleu_1,
            "bleu_2": avg_bleu_2,
            "rouge_1": avg_rouge_1,
            "rouge_2": avg_rouge_2,
            "rouge_l": avg_rouge_l,
            "bertscore_precision": bertscore_precision,
            "bertscore_recall": bertscore_recall,
            "bertscore_f1": bertscore_f1
        }
    }
    
    # Print results
    print("\n" + "=" * 60)
    print(f" EVALUATION RESULTS: {model_name}")
    print("=" * 60)
    print(f"BLEU:     {avg_bleu:.2f}")
    print(f"BLEU-1:   {avg_bleu_1:.2f}")
    print(f"BLEU-2:   {avg_bleu_2:.2f}")
    print(f"ROUGE-1:  {avg_rouge_1:.4f}")
    print(f"ROUGE-2:  {avg_rouge_2:.4f}")
    print(f"ROUGE-L:  {avg_rouge_l:.4f}")
    if include_bertscore:
        print(f"BERTScore P: {bertscore_precision:.4f}")
        print(f"BERTScore R: {bertscore_recall:.4f}")
        print(f"BERTScore F1: {bertscore_f1:.4f}")
    print("=" * 60)
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{model_name.lower().replace(' ', '_')}_eval.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n Results saved to: {output_file}")
    
    # Save sample generations
    samples_file = os.path.join(output_dir, f"{model_name.lower().replace(' ', '_')}_samples.json")
    samples = [
        {"prompt": p[:200], "reference": r[:200], "generated": g[:200]}
        for p, r, g in list(zip(prompts, references, predictions))[:10]
    ]
    with open(samples_file, "w") as f:
        json.dump(samples, f, indent=2)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned models")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--model_name", type=str, default="Model", help="Name for the model")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of test samples")
    parser.add_argument("--no_bertscore", action="store_true", help="Skip BERTScore calculation")
    parser.add_argument("--output_dir", type=str, default="results/evaluation")
    
    args = parser.parse_args()
    
    evaluate_model(
        model_path=args.model_path,
        model_name=args.model_name,
        num_samples=args.num_samples,
        include_bertscore=not args.no_bertscore,
        output_dir=args.output_dir
    )
