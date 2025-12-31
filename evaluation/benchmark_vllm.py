#!/usr/bin/env python3
"""
vLLM Inference Benchmark
========================
Benchmarks the fine-tuned model using vLLM for high-throughput inference.
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
import subprocess
import re


def get_gpu_memory_usage():
    """Get current GPU memory usage using nvidia-smi."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            memory_mb = int(result.stdout.strip().split('\n')[0])
            return memory_mb / 1024  # Convert to GB
    except Exception:
        pass
    return 0.0


# Test prompts (same as Transformers benchmark)
TEST_PROMPTS = [
    "### Instruction:\nExplain the concept of machine learning in simple terms.\n\n### Response:\n",
    "### Instruction:\nWrite a Python function to calculate the factorial of a number.\n\n### Response:\n",
    "### Instruction:\nWhat are the benefits of renewable energy sources?\n\n### Response:\n",
    "### Instruction:\nDescribe the water cycle in nature.\n\n### Response:\n",
    "### Instruction:\nHow does encryption protect data privacy?\n\n### Response:\n",
    "### Instruction:\nWhat is the difference between AI and machine learning?\n\n### Response:\n",
    "### Instruction:\nExplain how neural networks work.\n\n### Response:\n",
    "### Instruction:\nWhat are the main programming paradigms?\n\n### Response:\n",
    "### Instruction:\nDescribe the process of photosynthesis.\n\n### Response:\n",
    "### Instruction:\nHow does a blockchain work?\n\n### Response:\n",
    "### Instruction:\nWhat is the greenhouse effect?\n\n### Response:\n",
    "### Instruction:\nExplain the concept of object-oriented programming.\n\n### Response:\n",
    "### Instruction:\nWhat are the advantages of cloud computing?\n\n### Response:\n",
    "### Instruction:\nHow does GPS navigation work?\n\n### Response:\n",
    "### Instruction:\nWhat is quantum computing?\n\n### Response:\n",
]

def run_vllm_benchmark(
    model_path: str,
    num_trials: int = 5,
    max_new_tokens: int = 100,
    output_dir: str = "results/benchmarks"
):
    """Run vLLM inference benchmark."""
    
    print("=" * 60)
    print(" vLLM INFERENCE BENCHMARK")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Num trials: {num_trials}")
    print(f"Max new tokens: {max_new_tokens}")
    print(f"Num prompts: {len(TEST_PROMPTS)}")
    print("=" * 60)
    
    # Import vLLM
    from vllm import LLM, SamplingParams
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    # Load model with vLLM
    print("\n Loading model with vLLM...")
    load_start = time.time()
    
    try:
        llm = LLM(
            model=model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.8,
            max_model_len=2048,
            trust_remote_code=True,
            dtype="float16",
        )
        load_time = time.time() - load_start
        print(f"Model loaded in {load_time:.2f}s")
    except Exception as e:
        print(f" Error loading model: {e}")
        # Try with different settings
        print(f"Retrying with auto dtype...")
        llm = LLM(
            model=model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.8,
            max_model_len=2048,
            trust_remote_code=True,
            dtype="auto",
        )
        load_time = time.time() - load_start
        print(f"Model loaded in {load_time:.2f}s")
    
    # Sampling params
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.1,
    )
    
    # Warmup
    print("\n Warmup run...")
    _ = llm.generate([TEST_PROMPTS[0]], sampling_params)
    
    # Reset memory stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    # Benchmark runs
    print(f"\n  Running {num_trials} trials...")
    
    all_latencies = []
    all_throughputs = []
    all_tokens = []
    
    for trial in range(num_trials):
        print(f"\n   Trial {trial + 1}/{num_trials}:")
        trial_latencies = []
        trial_tokens = []
        
        for i, prompt in enumerate(TEST_PROMPTS):
            start_time = time.time()
            outputs = llm.generate([prompt], sampling_params)
            latency = time.time() - start_time
            
            output = outputs[0]
            tokens_generated = len(output.outputs[0].token_ids)
            throughput = tokens_generated / latency if latency > 0 else 0
            
            trial_latencies.append(latency)
            trial_tokens.append(tokens_generated)
            all_throughputs.append(throughput)
            
            print(f"Prompt {i+1}: {latency:.3f}s, {tokens_generated} tokens, {throughput:.1f} tok/s")
        
        all_latencies.extend(trial_latencies)
        all_tokens.extend(trial_tokens)
    
    # Get peak memory using nvidia-smi (vLLM runs in separate process)
    peak_memory_gb = get_gpu_memory_usage()
    print(f"\n📊 GPU Memory used: {peak_memory_gb:.2f} GB")
    
    # Calculate statistics
    import statistics
    
    avg_latency = statistics.mean(all_latencies)
    std_latency = statistics.stdev(all_latencies) if len(all_latencies) > 1 else 0
    avg_tokens = statistics.mean(all_tokens)
    avg_throughput = statistics.mean(all_throughputs)
    std_throughput = statistics.stdev(all_throughputs) if len(all_throughputs) > 1 else 0
    
    # Results
    results = {
        "framework": "vllm",
        "timestamp": datetime.now().isoformat(),
        "model_path": model_path,
        "config": {
            "num_prompts": len(TEST_PROMPTS),
            "num_trials": num_trials,
            "max_new_tokens": max_new_tokens
        },
        "results": {
            "avg_latency_s": avg_latency,
            "std_latency_s": std_latency,
            "avg_tokens_generated": avg_tokens,
            "avg_throughput_tokens_per_s": avg_throughput,
            "std_throughput": std_throughput,
            "peak_memory_gb": peak_memory_gb,
            "load_time_s": load_time
        }
    }
    
    # Print summary
    print("\n" + "=" * 60)
    print(" VLLM BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Average latency: {avg_latency:.3f}s (±{std_latency:.3f})")
    print(f"Average tokens: {avg_tokens:.1f}")
    print(f"Throughput: {avg_throughput:.2f} tokens/s (±{std_throughput:.2f})")
    print(f"Peak memory: {peak_memory_gb:.2f} GB")
    print(f"Load time: {load_time:.2f}s")
    print("=" * 60)
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "vllm_benchmark.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n Results saved to: {output_path}")
    
    # Generate sample outputs
    print("\n Sample generation:")
    print("-" * 40)
    sample_output = llm.generate([TEST_PROMPTS[0]], sampling_params)[0]
    print(f"Prompt: {TEST_PROMPTS[0][:50]}...")
    print(f"Response: {sample_output.outputs[0].text[:200]}...")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="vLLM Inference Benchmark")
    parser.add_argument(
        "--model_path",
        type=str,
        default="outputs/transformers/20251227_134936/merged_model",
        help="Path to the model"
    )
    parser.add_argument("--num_trials", type=int, default=3)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="results/benchmarks")
    
    args = parser.parse_args()
    
    run_vllm_benchmark(
        model_path=args.model_path,
        num_trials=args.num_trials,
        max_new_tokens=args.max_new_tokens,
        output_dir=args.output_dir
    )
