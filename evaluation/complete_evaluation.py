#!/usr/bin/env python3
"""
Complete Project Evaluation
===========================
Runs all missing evaluations:
1. BERTScore (with smaller model)
2. Before vs After comparison (base model vs fine-tuned)
3. Unsloth inference benchmark
4. Human evaluation setup
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from datasets import load_dataset
from tqdm import tqdm


def clear_gpu():
    """Clear GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def load_test_samples(num_samples: int = 50):
    """Load test samples."""
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
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


def generate_with_model(model_path: str, prompts: list, max_new_tokens: int = 128, model_name: str = "Model"):
    """Generate responses using a model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    
    print(f"\n Loading {model_name}: {model_path}")
    
    # Check if it's the base model (from HuggingFace) or local
    is_local = os.path.exists(model_path)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    if is_local:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        # Base model - use quantization to save memory
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    generations = []
    print(f"Generating {len(prompts)} responses...")
    
    for prompt in tqdm(prompts, desc=f"Generating ({model_name})"):
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
        response = generated_text[len(prompt):].strip()
        generations.append(response)
    
    del model
    clear_gpu()
    
    return generations


def calculate_bertscore_light(predictions: list, references: list):
    """Calculate BERTScore with a lighter model."""
    try:
        from bert_score import score as bert_score
        
        print("\n Computing BERTScore (using distilbert-base-uncased)...")
        P, R, F1 = bert_score(
            predictions,
            references,
            model_type="distilbert-base-uncased",  # Much lighter!
            batch_size=32,
            verbose=False,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        return {
            "bertscore_precision": P.mean().item(),
            "bertscore_recall": R.mean().item(),
            "bertscore_f1": F1.mean().item()
        }
    except Exception as e:
        print(f"⚠ BERTScore failed: {e}")
        return {"bertscore_precision": 0, "bertscore_recall": 0, "bertscore_f1": 0}


def calculate_metrics(predictions: list, references: list):
    """Calculate BLEU and ROUGE."""
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.tokenize import word_tokenize
    from rouge_score import rouge_scorer
    import nltk
    
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except:
        nltk.download('punkt_tab', quiet=True)
    
    smoothing = SmoothingFunction()
    bleu_scores = []
    
    for pred, ref in zip(predictions, references):
        pred_tokens = word_tokenize(pred.lower())
        ref_tokens = [word_tokenize(ref.lower())]
        try:
            bleu = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smoothing.method1)
        except:
            bleu = 0.0
        bleu_scores.append(bleu)
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_l_scores = []
    
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        rouge_l_scores.append(scores['rougeL'].fmeasure)
    
    return {
        "bleu": sum(bleu_scores) / len(bleu_scores) * 100,
        "rouge_l": sum(rouge_l_scores) / len(rouge_l_scores)
    }


def run_before_after_comparison():
    """Compare base model (before) vs fine-tuned (after)."""
    print("\n" + "=" * 70)
    print(" BEFORE VS AFTER FINE-TUNING COMPARISON")
    print("=" * 70)
    
    # Load test data
    prompts, references = load_test_samples(30)
    
    results = {}
    
    # 1. Base model (BEFORE fine-tuning)
    print("\n🔹 Evaluating BASE MODEL (before fine-tuning)...")
    base_generations = generate_with_model(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        prompts,
        model_name="Base TinyLlama"
    )
    base_metrics = calculate_metrics(base_generations, references)
    results["base_model"] = base_metrics
    print(f"BLEU: {base_metrics['bleu']:.2f}, ROUGE-L: {base_metrics['rouge_l']:.4f}")
    
    clear_gpu()
    
    # 2. Fine-tuned model (AFTER fine-tuning) - Transformers
    print("\n🔹 Evaluating FINE-TUNED MODEL (Transformers+PEFT)...")
    ft_trans_generations = generate_with_model(
        "outputs/transformers/20251227_134936/merged_model",
        prompts,
        model_name="Fine-tuned Transformers"
    )
    ft_trans_metrics = calculate_metrics(ft_trans_generations, references)
    results["finetuned_transformers"] = ft_trans_metrics
    print(f"BLEU: {ft_trans_metrics['bleu']:.2f}, ROUGE-L: {ft_trans_metrics['rouge_l']:.4f}")
    
    clear_gpu()
    
    # 3. Fine-tuned model (AFTER fine-tuning) - Unsloth
    print("\n🔹 Evaluating FINE-TUNED MODEL (Unsloth)...")
    ft_unsloth_generations = generate_with_model(
        "outputs/unsloth/20251227_160819/merged_model",
        prompts,
        model_name="Fine-tuned Unsloth"
    )
    ft_unsloth_metrics = calculate_metrics(ft_unsloth_generations, references)
    results["finetuned_unsloth"] = ft_unsloth_metrics
    print(f"BLEU: {ft_unsloth_metrics['bleu']:.2f}, ROUGE-L: {ft_unsloth_metrics['rouge_l']:.4f}")
    
    # Summary
    print("\n" + "=" * 70)
    print(" BEFORE VS AFTER COMPARISON RESULTS")
    print("=" * 70)
    print(f"{'Model':<30} {'BLEU':>10} {'ROUGE-L':>10} {'Improvement':>15}")
    print("-" * 70)
    print(f"{'Base TinyLlama (Before)':<30} {base_metrics['bleu']:>10.2f} {base_metrics['rouge_l']:>10.4f} {'--':>15}")
    
    trans_bleu_imp = ft_trans_metrics['bleu'] - base_metrics['bleu']
    trans_rouge_imp = ft_trans_metrics['rouge_l'] - base_metrics['rouge_l']
    print(f"{'Fine-tuned Transformers':<30} {ft_trans_metrics['bleu']:>10.2f} {ft_trans_metrics['rouge_l']:>10.4f} {f'+{trans_bleu_imp:.2f} BLEU':>15}")
    
    unsloth_bleu_imp = ft_unsloth_metrics['bleu'] - base_metrics['bleu']
    unsloth_rouge_imp = ft_unsloth_metrics['rouge_l'] - base_metrics['rouge_l']
    print(f"{'Fine-tuned Unsloth':<30} {ft_unsloth_metrics['bleu']:>10.2f} {ft_unsloth_metrics['rouge_l']:>10.4f} {f'+{unsloth_bleu_imp:.2f} BLEU':>15}")
    print("=" * 70)
    
    # Save results
    results["timestamp"] = datetime.now().isoformat()
    results["num_samples"] = len(prompts)
    
    os.makedirs("results/evaluation", exist_ok=True)
    with open("results/evaluation/before_after_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n Results saved to: results/evaluation/before_after_comparison.json")
    
    return results


def run_unsloth_inference_benchmark():
    """Benchmark Unsloth inference."""
    print("\n" + "=" * 70)
    print(" UNSLOTH INFERENCE BENCHMARK")
    print("=" * 70)
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model_path = "outputs/unsloth/20251227_160819/merged_model"
    
    print(f"Model: {model_path}")
    
    # Test prompts
    prompts = [
        "### Instruction:\nExplain machine learning.\n\n### Response:\n",
        "### Instruction:\nWrite a factorial function in Python.\n\n### Response:\n",
        "### Instruction:\nWhat are renewable energy benefits?\n\n### Response:\n",
        "### Instruction:\nDescribe the water cycle.\n\n### Response:\n",
        "### Instruction:\nHow does encryption work?\n\n### Response:\n",
    ]
    
    print(f"\n Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Warmup
    print(" Warmup...")
    inputs = tokenizer(prompts[0], return_tensors="pt").to(model.device)
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=50)
    
    # Reset memory stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    # Benchmark
    print(f"\n  Running benchmark (3 trials x {len(prompts)} prompts)...")
    
    latencies = []
    throughputs = []
    tokens_list = []
    
    for trial in range(3):
        print(f"\n   Trial {trial + 1}/3:")
        for i, prompt in enumerate(prompts):
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            start_time = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True
                )
            latency = time.time() - start_time
            
            tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
            throughput = tokens / latency if latency > 0 else 0
            
            latencies.append(latency)
            throughputs.append(throughput)
            tokens_list.append(tokens)
            
            print(f"Prompt {i+1}: {latency:.3f}s, {tokens} tokens, {throughput:.1f} tok/s")
    
    # Peak memory
    peak_memory = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
    
    # Results
    import statistics
    results = {
        "framework": "unsloth",
        "timestamp": datetime.now().isoformat(),
        "model_path": model_path,
        "config": {
            "num_prompts": len(prompts),
            "num_trials": 3,
            "max_new_tokens": 100
        },
        "results": {
            "avg_latency_s": statistics.mean(latencies),
            "std_latency_s": statistics.stdev(latencies),
            "avg_tokens_generated": statistics.mean(tokens_list),
            "avg_throughput_tokens_per_s": statistics.mean(throughputs),
            "std_throughput": statistics.stdev(throughputs),
            "peak_memory_gb": peak_memory
        }
    }
    
    print("\n" + "=" * 60)
    print(" UNSLOTH INFERENCE RESULTS")
    print("=" * 60)
    print(f"Average latency: {results['results']['avg_latency_s']:.3f}s")
    print(f"Throughput: {results['results']['avg_throughput_tokens_per_s']:.2f} tokens/s")
    print(f"Peak memory: {peak_memory:.2f} GB")
    print("=" * 60)
    
    # Save
    os.makedirs("results/benchmarks", exist_ok=True)
    with open("results/benchmarks/unsloth_benchmark.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n Results saved to: results/benchmarks/unsloth_benchmark.json")
    
    del model
    clear_gpu()
    
    return results


def generate_human_eval_samples():
    """Generate samples for human evaluation."""
    print("\n" + "=" * 70)
    print(" HUMAN EVALUATION SAMPLES")
    print("=" * 70)
    
    # 5 random prompts from test set
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
    test_start = len(dataset) - 1000
    
    import random
    random.seed(42)
    sample_indices = random.sample(range(test_start, len(dataset)), 5)
    
    samples = []
    for idx in sample_indices:
        sample = dataset[idx]
        instruction = sample["instruction"]
        context = sample.get("context", "")
        reference = sample["response"]
        
        if context:
            prompt = f"### Instruction:\n{instruction}\n\n### Context:\n{context}\n\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
        
        samples.append({
            "id": len(samples) + 1,
            "instruction": instruction,
            "context": context,
            "prompt": prompt,
            "reference": reference,
            "generated_transformers": "",
            "generated_unsloth": "",
            "eval_transformers": {
                "helpfulness": 0,
                "factuality": 0,
                "instruction_following": 0
            },
            "eval_unsloth": {
                "helpfulness": 0,
                "factuality": 0,
                "instruction_following": 0
            }
        })
    
    # Generate responses
    print("\n Generating responses with fine-tuned models...")
    
    prompts = [s["prompt"] for s in samples]
    
    # Transformers
    trans_responses = generate_with_model(
        "outputs/transformers/20251227_134936/merged_model",
        prompts,
        model_name="Transformers"
    )
    for i, resp in enumerate(trans_responses):
        samples[i]["generated_transformers"] = resp
    
    clear_gpu()
    
    # Unsloth
    unsloth_responses = generate_with_model(
        "outputs/unsloth/20251227_160819/merged_model",
        prompts,
        model_name="Unsloth"
    )
    for i, resp in enumerate(unsloth_responses):
        samples[i]["generated_unsloth"] = resp
    
    # Save
    os.makedirs("results/evaluation", exist_ok=True)
    with open("results/evaluation/human_eval_samples.json", "w") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    
    # Print for manual evaluation
    print("\n" + "=" * 70)
    print(" HUMAN EVALUATION FORM")
    print("Rate each response on a scale of 1-5:")
    print(f"1 = Very Poor, 2 = Poor, 3 = Average, 4 = Good, 5 = Excellent")
    print("=" * 70)
    
    for i, sample in enumerate(samples):
        print(f"\n{'='*70}")
        print(f"SAMPLE {i+1}/5")
        print(f"{'='*70}")
        print(f"\n INSTRUCTION:\n{sample['instruction'][:300]}...")
        print(f"\n REFERENCE:\n{sample['reference'][:300]}...")
        print(f"\n TRANSFORMERS RESPONSE:\n{sample['generated_transformers'][:300]}...")
        print(f"\n UNSLOTH RESPONSE:\n{sample['generated_unsloth'][:300]}...")
        print("\n" + "-"*70)
    
    print(f"\n Full samples saved to: results/evaluation/human_eval_samples.json")
    print(f"Please fill in the ratings in the JSON file.")
    
    return samples


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Run all evaluations")
    parser.add_argument("--before_after", action="store_true", help="Before vs After comparison")
    parser.add_argument("--unsloth_bench", action="store_true", help="Unsloth inference benchmark")
    parser.add_argument("--human_eval", action="store_true", help="Generate human eval samples")
    
    args = parser.parse_args()
    
    if args.all or args.before_after:
        run_before_after_comparison()
    
    if args.all or args.unsloth_bench:
        run_unsloth_inference_benchmark()
    
    if args.all or args.human_eval:
        generate_human_eval_samples()
    
    if not any([args.all, args.before_after, args.unsloth_bench, args.human_eval]):
        print("Usage: python complete_evaluation.py --all")
        print(f"python complete_evaluation.py --before_after")
        print(f"python complete_evaluation.py --unsloth_bench")
        print(f"python complete_evaluation.py --human_eval")
