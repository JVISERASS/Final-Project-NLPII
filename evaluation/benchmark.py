"""
NLP2 Final Project - Inference Benchmarking
============================================
Benchmarks inference performance across all 4 frameworks:
- Transformers
- Unsloth
- vLLM
- Ollama

Metrics:
- Latency per request (seconds)
- Throughput (tokens/second)
- Peak memory (GB)

Optimized for RTX 4070 SUPER (12GB VRAM)
"""

import os
import sys
import json
import time
import gc
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

import torch
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.inference_engines import (
    get_inference_engine,
    TransformersInference,
    UnslothInference,
    VLLMInference,
    OllamaInference,
    BenchmarkResult
)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


@dataclass
class FrameworkBenchmark:
    """Results for a single framework benchmark."""
    framework: str
    model_path: str
    
    # Latency metrics
    avg_latency_seconds: float
    min_latency_seconds: float
    max_latency_seconds: float
    p50_latency_seconds: float
    p90_latency_seconds: float
    p99_latency_seconds: float
    
    # Throughput metrics
    avg_throughput_tokens_per_sec: float
    total_tokens_generated: int
    
    # Memory metrics
    peak_memory_gb: float
    avg_memory_gb: float
    
    # Test configuration
    num_prompts: int
    num_warmup_runs: int
    max_new_tokens: int
    temperature: float
    
    # Timing
    total_benchmark_time_seconds: float
    timestamp: str
    
    def to_dict(self) -> Dict:
        return asdict(self)


class InferenceBenchmarker:
    """
    Comprehensive benchmarking suite for inference frameworks.
    
    Tests latency, throughput, and memory usage across:
    - Different frameworks (Transformers, Unsloth, vLLM, Ollama)
    - Different input lengths
    - Different output lengths
    - Batched vs single inference
    """
    
    def __init__(
        self,
        config: Optional[dict] = None,
        config_path: str = "configs/config.yaml"
    ):
        self.config = config or load_config(config_path)
        self.results = {}
        
        # Benchmark configuration
        self.benchmark_config = self.config.get("benchmark", {})
        self.num_warmup = self.benchmark_config.get("num_warmup_runs", 3)
        self.num_runs = self.benchmark_config.get("num_benchmark_runs", 10)
        
        # Inference configuration
        self.inference_config = self.config.get("inference", {})
        
    def _clear_gpu_memory(self):
        """Clear GPU memory between runs."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    
    def _get_test_prompts(
        self,
        num_prompts: int = 10,
        prompt_file: Optional[str] = None
    ) -> List[str]:
        """Get test prompts for benchmarking."""
        
        if prompt_file and Path(prompt_file).exists():
            with open(prompt_file, "r") as f:
                data = [json.loads(line) for line in f]
            return [d["prompt"] for d in data[:num_prompts]]
        
        # Default test prompts with varying complexity
        default_prompts = [
            # Short prompts
            "<|system|>\nYou are a helpful assistant.</s>\n<|user|>\nWhat is machine learning?</s>\n<|assistant|>\n",
            "<|system|>\nYou are a helpful assistant.</s>\n<|user|>\nExplain Python in one sentence.</s>\n<|assistant|>\n",
            "<|system|>\nYou are a helpful assistant.</s>\n<|user|>\nWhat is 2+2?</s>\n<|assistant|>\n",
            
            # Medium prompts
            "<|system|>\nYou are a helpful assistant.</s>\n<|user|>\nExplain the difference between supervised and unsupervised learning in machine learning.</s>\n<|assistant|>\n",
            "<|system|>\nYou are a helpful assistant.</s>\n<|user|>\nWrite a simple Python function that calculates the factorial of a number.</s>\n<|assistant|>\n",
            "<|system|>\nYou are a helpful assistant.</s>\n<|user|>\nWhat are the main components of a neural network and how do they work together?</s>\n<|assistant|>\n",
            
            # Longer prompts with context
            "<|system|>\nYou are a helpful assistant.</s>\n<|user|>\nI'm working on a data science project and need to preprocess my dataset. The dataset contains missing values, outliers, and categorical variables. What steps should I follow to clean and prepare this data for a machine learning model?</s>\n<|assistant|>\n",
            "<|system|>\nYou are a helpful assistant.</s>\n<|user|>\nExplain the attention mechanism in transformers. How does self-attention work and why is it important for natural language processing tasks?</s>\n<|assistant|>\n",
            "<|system|>\nYou are a helpful assistant.</s>\n<|user|>\nCompare and contrast different types of neural network architectures: CNNs, RNNs, and Transformers. What are each best suited for?</s>\n<|assistant|>\n",
            "<|system|>\nYou are a helpful assistant.</s>\n<|user|>\nI want to deploy a machine learning model to production. What are the best practices for MLOps, including model versioning, monitoring, and CI/CD pipelines?</s>\n<|assistant|>\n",
        ]
        
        # Repeat if needed
        while len(default_prompts) < num_prompts:
            default_prompts.extend(default_prompts)
        
        return default_prompts[:num_prompts]
    
    def benchmark_framework(
        self,
        framework: str,
        model_path: str,
        prompts: List[str],
        num_warmup: Optional[int] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs
    ) -> FrameworkBenchmark:
        """
        Benchmark a single framework.
        
        Args:
            framework: Name of framework (transformers, unsloth, vllm, ollama)
            model_path: Path to model or model name
            prompts: List of test prompts
            num_warmup: Number of warmup runs
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            FrameworkBenchmark with results
        """
        num_warmup = num_warmup or self.num_warmup
        
        print(f"\n{'='*60}")
        print(f"BENCHMARKING: {framework.upper()}")
        print(f"Model: {model_path}")
        print(f"Prompts: {len(prompts)}")
        print(f"Max new tokens: {max_new_tokens}")
        print(f"{'='*60}")
        
        # Clear memory
        self._clear_gpu_memory()
        
        # Load engine
        print(f"\n Loading {framework} inference engine...")
        engine = get_inference_engine(framework)
        
        try:
            if framework == "ollama":
                engine.load_model(model_path, **kwargs)
            elif framework == "vllm":
                engine.load_model(model_path, **kwargs)
            else:
                engine.load_model(model_path, **kwargs)
        except Exception as e:
            print(f" Failed to load model: {e}")
            return None
        
        # Warmup
        print(f"\n Warming up ({num_warmup} runs)...")
        for i in range(num_warmup):
            _ = engine.generate(
                prompts[i % len(prompts)],
                max_new_tokens=max_new_tokens,
                temperature=temperature
            )
        
        # Reset memory stats after warmup
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        # Benchmark runs
        print(f"\n  Running benchmark ({len(prompts)} prompts)...")
        
        latencies = []
        throughputs = []
        memory_readings = []
        total_tokens = 0
        
        benchmark_start = time.time()
        
        for i, prompt in enumerate(tqdm(prompts, desc="Benchmarking")):
            result = engine.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature
            )
            
            latencies.append(result.latency_seconds)
            throughputs.append(result.tokens_per_second)
            total_tokens += result.tokens_generated
            
            if torch.cuda.is_available():
                memory_readings.append(torch.cuda.memory_allocated() / 1024**3)
        
        benchmark_time = time.time() - benchmark_start
        
        # Calculate statistics
        peak_memory = 0
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / 1024**3
        
        avg_memory = np.mean(memory_readings) if memory_readings else 0
        
        result = FrameworkBenchmark(
            framework=framework,
            model_path=model_path,
            
            # Latency
            avg_latency_seconds=np.mean(latencies),
            min_latency_seconds=np.min(latencies),
            max_latency_seconds=np.max(latencies),
            p50_latency_seconds=np.percentile(latencies, 50),
            p90_latency_seconds=np.percentile(latencies, 90),
            p99_latency_seconds=np.percentile(latencies, 99),
            
            # Throughput
            avg_throughput_tokens_per_sec=np.mean(throughputs),
            total_tokens_generated=total_tokens,
            
            # Memory
            peak_memory_gb=peak_memory,
            avg_memory_gb=avg_memory,
            
            # Config
            num_prompts=len(prompts),
            num_warmup_runs=num_warmup,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            
            # Timing
            total_benchmark_time_seconds=benchmark_time,
            timestamp=datetime.now().isoformat()
        )
        
        # Cleanup
        engine.cleanup()
        self._clear_gpu_memory()
        
        # Print results
        self._print_benchmark_result(result)
        
        return result
    
    def _print_benchmark_result(self, result: FrameworkBenchmark):
        """Print benchmark results."""
        print(f"\n Results for {result.framework.upper()}:")
        print(f"{'='*50}")
        
        print(f"\n   Latency:")
        print(f"Average:  {result.avg_latency_seconds:.3f}s")
        print(f"Min:      {result.min_latency_seconds:.3f}s")
        print(f"Max:      {result.max_latency_seconds:.3f}s")
        print(f"P50:      {result.p50_latency_seconds:.3f}s")
        print(f"P90:      {result.p90_latency_seconds:.3f}s")
        print(f"P99:      {result.p99_latency_seconds:.3f}s")
        
        print(f"\n   Throughput:")
        print(f"Average:  {result.avg_throughput_tokens_per_sec:.1f} tokens/sec")
        print(f"Total:    {result.total_tokens_generated:,} tokens")
        
        print(f"\n   Memory:")
        print(f"Peak:     {result.peak_memory_gb:.2f} GB")
        print(f"Average:  {result.avg_memory_gb:.2f} GB")
        
        print(f"\n   Benchmark Time: {result.total_benchmark_time_seconds:.1f}s")
    
    def run_all_benchmarks(
        self,
        models: Dict[str, str],
        num_prompts: int = 20,
        prompt_file: Optional[str] = None,
        **kwargs
    ) -> Dict[str, FrameworkBenchmark]:
        """
        Run benchmarks on all frameworks.
        
        Args:
            models: Dictionary mapping framework name to model path
                   e.g., {"transformers": "outputs/transformers/merged_model", ...}
            num_prompts: Number of prompts to test
            prompt_file: Optional file with test prompts
            
        Returns:
            Dictionary of framework name to FrameworkBenchmark
        """
        prompts = self._get_test_prompts(num_prompts, prompt_file)
        
        print("\n" + "="*70)
        print(" RUNNING COMPLETE BENCHMARK SUITE")
        print("="*70)
        print(f"Frameworks: {list(models.keys())}")
        print(f"Prompts: {len(prompts)}")
        print(f"Warmup runs: {self.num_warmup}")
        
        results = {}
        
        for framework, model_path in models.items():
            try:
                result = self.benchmark_framework(
                    framework=framework,
                    model_path=model_path,
                    prompts=prompts,
                    **kwargs
                )
                if result:
                    results[framework] = result
            except Exception as e:
                print(f" Error benchmarking {framework}: {e}")
                continue
        
        self.results = results
        return results
    
    def create_comparison_table(self, results: Optional[Dict[str, FrameworkBenchmark]] = None) -> pd.DataFrame:
        """Create comparison table of all frameworks."""
        results = results or self.results
        
        data = []
        for name, result in results.items():
            data.append({
                "Framework": name,
                "Avg Latency (s)": f"{result.avg_latency_seconds:.3f}",
                "P90 Latency (s)": f"{result.p90_latency_seconds:.3f}",
                "Throughput (tok/s)": f"{result.avg_throughput_tokens_per_sec:.1f}",
                "Peak Memory (GB)": f"{result.peak_memory_gb:.2f}",
                "Total Tokens": result.total_tokens_generated,
            })
        
        df = pd.DataFrame(data)
        return df
    
    def save_results(
        self,
        results: Optional[Dict[str, FrameworkBenchmark]] = None,
        output_dir: str = "results"
    ):
        """Save benchmark results to files."""
        results = results or self.results
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save raw results
        raw_results = {name: result.to_dict() for name, result in results.items()}
        with open(output_path / f"benchmark_results_{timestamp}.json", "w") as f:
            json.dump(raw_results, f, indent=2)
        
        # Save comparison table
        df = self.create_comparison_table(results)
        df.to_csv(output_path / f"benchmark_comparison_{timestamp}.csv", index=False)
        
        print(f"\n Results saved to {output_path}")
        print(f"- benchmark_results_{timestamp}.json")
        print(f"- benchmark_comparison_{timestamp}.csv")
        
        return output_path
    
    def print_summary(self, results: Optional[Dict[str, FrameworkBenchmark]] = None):
        """Print summary comparison of all frameworks."""
        results = results or self.results
        
        print("\n" + "="*80)
        print(" BENCHMARK SUMMARY - ALL FRAMEWORKS")
        print("="*80)
        
        df = self.create_comparison_table(results)
        print(df.to_string(index=False))
        
        # Find winners
        print("\n WINNERS:")
        
        # Fastest (lowest latency)
        fastest = min(results.items(), key=lambda x: x[1].avg_latency_seconds)
        print(f"Fastest (latency): {fastest[0]} ({fastest[1].avg_latency_seconds:.3f}s)")
        
        # Highest throughput
        highest_throughput = max(results.items(), key=lambda x: x[1].avg_throughput_tokens_per_sec)
        print(f"Highest throughput: {highest_throughput[0]} ({highest_throughput[1].avg_throughput_tokens_per_sec:.1f} tok/s)")
        
        # Most memory efficient
        most_efficient = min(results.items(), key=lambda x: x[1].peak_memory_gb)
        print(f"Most memory efficient: {most_efficient[0]} ({most_efficient[1].peak_memory_gb:.2f} GB)")
        
        print("\n" + "="*80)


def main():
    """Run benchmark suite."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark inference frameworks")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--num_prompts", type=int, default=20)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--transformers_model", type=str, default=None)
    parser.add_argument("--unsloth_model", type=str, default=None)
    parser.add_argument("--vllm_model", type=str, default=None)
    parser.add_argument("--ollama_model", type=str, default=None)
    
    args = parser.parse_args()
    
    # Initialize benchmarker
    benchmarker = InferenceBenchmarker(config_path=args.config)
    
    # Define models to benchmark
    models = {}
    
    if args.transformers_model:
        models["transformers"] = args.transformers_model
    if args.unsloth_model:
        models["unsloth"] = args.unsloth_model
    if args.vllm_model:
        models["vllm"] = args.vllm_model
    if args.ollama_model:
        models["ollama"] = args.ollama_model
    
    # If no models specified, use defaults
    if not models:
        print("⚠  No models specified. Use --transformers_model, --unsloth_model, etc.")
        print(f"Example: python benchmark.py --transformers_model outputs/transformers/merged_model")
        return
    
    # Run benchmarks
    results = benchmarker.run_all_benchmarks(
        models=models,
        num_prompts=args.num_prompts,
        max_new_tokens=args.max_tokens
    )
    
    # Print summary
    benchmarker.print_summary()
    
    # Save results
    benchmarker.save_results()


if __name__ == "__main__":
    main()
