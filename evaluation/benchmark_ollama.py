#!/usr/bin/env python3
"""
Ollama Inference Benchmark
==========================
Benchmarks the fine-tuned model using Ollama for local model serving.
"""

import os
import sys
import json
import time
import subprocess
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


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


# Test prompts (same as other benchmarks)
TEST_PROMPTS = [
    "Explain the concept of machine learning in simple terms.",
    "Write a Python function to calculate the factorial of a number.",
    "What are the benefits of renewable energy sources?",
    "Describe the water cycle in nature.",
    "How does encryption protect data privacy?",
    "What is the difference between AI and machine learning?",
    "Explain how neural networks work.",
    "What are the main programming paradigms?",
    "Describe the process of photosynthesis.",
    "How does a blockchain work?",
    "What is the greenhouse effect?",
    "Explain the concept of object-oriented programming.",
    "What are the advantages of cloud computing?",
    "How does GPS navigation work?",
    "What is quantum computing?",
]

def run_ollama_benchmark(
    model_name: str = "tinyllama-finetuned",
    num_trials: int = 5,
    max_new_tokens: int = 100,
    output_dir: str = "results/benchmarks"
):
    """Run Ollama inference benchmark."""
    
    print("=" * 60)
    print(" OLLAMA INFERENCE BENCHMARK")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Num trials: {num_trials}")
    print(f"Max new tokens: ~{max_new_tokens}")
    print(f"Num prompts: {len(TEST_PROMPTS)}")
    print("=" * 60)
    
    # Check if model exists
    result = subprocess.run(
        ["ollama", "list"],
        capture_output=True,
        text=True
    )
    if model_name not in result.stdout:
        print(f" Model {model_name} not found in Ollama")
        print("Available models:", result.stdout)
        return None
    
    print(f"\nModel {model_name} found in Ollama")
    
    # Warmup - load model
    print("\n Warming up (loading model)...")
    warmup_start = time.time()
    warmup_result = subprocess.run(
        ["ollama", "run", model_name, "Hello"],
        capture_output=True,
        text=True,
        timeout=120
    )
    load_time = time.time() - warmup_start
    print(f"Model loaded in {load_time:.2f}s")
    
    # Benchmark runs
    print(f"\n  Running {num_trials} trials...")
    
    all_latencies = []
    all_throughputs = []
    all_tokens = []
    
    for trial in range(num_trials):
        print(f"\n   Trial {trial + 1}/{num_trials}:")
        
        for i, prompt in enumerate(TEST_PROMPTS):
            # Format prompt as instruction
            formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:"
            
            start_time = time.time()
            
            try:
                result = subprocess.run(
                    ["ollama", "run", model_name, formatted_prompt],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                latency = time.time() - start_time
                
                output_text = result.stdout.strip()
                # Estimate tokens (rough approximation: ~4 chars per token)
                tokens_generated = len(output_text) // 4
                tokens_generated = max(1, min(tokens_generated, max_new_tokens))
                
                throughput = tokens_generated / latency if latency > 0 else 0
                
                all_latencies.append(latency)
                all_tokens.append(tokens_generated)
                all_throughputs.append(throughput)
                
                print(f"Prompt {i+1}: {latency:.3f}s, ~{tokens_generated} tokens, {throughput:.1f} tok/s")
                
            except subprocess.TimeoutExpired:
                print(f"Prompt {i+1}: TIMEOUT")
                all_latencies.append(60.0)
                all_tokens.append(0)
                all_throughputs.append(0)
    
    # Calculate statistics
    import statistics
    
    # Filter out timeouts for statistics
    valid_latencies = [l for l in all_latencies if l < 60]
    valid_throughputs = [t for t in all_throughputs if t > 0]
    valid_tokens = [t for t in all_tokens if t > 0]
    
    if not valid_latencies:
        print(" All requests timed out")
        return None
    
    avg_latency = statistics.mean(valid_latencies)
    std_latency = statistics.stdev(valid_latencies) if len(valid_latencies) > 1 else 0
    avg_tokens = statistics.mean(valid_tokens) if valid_tokens else 0
    avg_throughput = statistics.mean(valid_throughputs) if valid_throughputs else 0
    std_throughput = statistics.stdev(valid_throughputs) if len(valid_throughputs) > 1 else 0
    
    # Get GPU memory usage
    peak_memory_gb = get_gpu_memory_usage()
    print(f"\n📊 GPU Memory used: {peak_memory_gb:.2f} GB")
    
    # Results
    results = {
        "framework": "ollama",
        "timestamp": datetime.now().isoformat(),
        "model_name": model_name,
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
    print(" OLLAMA BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Average latency: {avg_latency:.3f}s (±{std_latency:.3f})")
    print(f"Average tokens: {avg_tokens:.1f}")
    print(f"Throughput: {avg_throughput:.2f} tokens/s (±{std_throughput:.2f})")
    print(f"Load time: {load_time:.2f}s")
    print("=" * 60)
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "ollama_benchmark.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n Results saved to: {output_path}")
    
    # Generate sample output
    print("\n Sample generation:")
    print("-" * 40)
    sample_prompt = f"### Instruction:\n{TEST_PROMPTS[0]}\n\n### Response:"
    sample_result = subprocess.run(
        ["ollama", "run", model_name, sample_prompt],
        capture_output=True,
        text=True,
        timeout=60
    )
    print(f"Prompt: {TEST_PROMPTS[0][:50]}...")
    print(f"Response: {sample_result.stdout[:200]}...")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Ollama Inference Benchmark")
    parser.add_argument(
        "--model_name",
        type=str,
        default="tinyllama-finetuned",
        help="Name of the Ollama model"
    )
    parser.add_argument("--num_trials", type=int, default=3)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="results/benchmarks")
    
    args = parser.parse_args()
    
    run_ollama_benchmark(
        model_name=args.model_name,
        num_trials=args.num_trials,
        max_new_tokens=args.max_new_tokens,
        output_dir=args.output_dir
    )
