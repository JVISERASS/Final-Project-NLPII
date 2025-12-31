#!/usr/bin/env python3
"""
Main Pipeline Script - Fine-tune Once, Serve Anywhere
======================================================
NLP II Final Project - Complete Orchestration Pipeline

This script provides a unified interface to run all project components:
1. Data preparation
2. Fine-tuning (Transformers+PEFT or Unsloth)
3. Model evaluation (BLEU, ROUGE, BERTScore)
4. Inference benchmarking (4 frameworks)
5. Results visualization

Usage:
    python main.py --help                               # Show help
    python main.py --mode full                          # Run complete pipeline
    python main.py --mode data                          # Only data preparation
    python main.py --mode train --trainer transformers  # Train with Transformers+PEFT
    python main.py --mode train --trainer unsloth       # Train with Unsloth
    python main.py --mode evaluate                      # Only evaluation
    python main.py --mode benchmark                     # Only benchmarking
    python main.py --mode visualize                     # Generate visualizations
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def print_header(text: str):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_step(step_num: int, text: str):
    """Print a step indicator."""
    print(f"\n[Step {step_num}] {text}")
    print("-" * 50)


def print_gpu_info():
    """Print GPU information."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch Version: {torch.__version__}")
    else:
        print("WARNING: No GPU available. Running on CPU.")


# ============================================================================
# STEP 1: DATA PREPARATION
# ============================================================================

def run_data_preparation(config: dict, output_dir: Path) -> Dict[str, Any]:
    """Prepare the Dolly dataset for training."""
    print_step(1, "Data Preparation")
    
    from data.prepare_data import DollyDataProcessor
    
    processor = DollyDataProcessor(config=config)
    formatted_dataset, eval_dataset = processor.prepare_all(
        save=True,
        output_dir="data/processed"
    )
    
    results = {
        "train_samples": len(formatted_dataset["train"]),
        "val_samples": len(formatted_dataset["validation"]),
        "test_samples": len(formatted_dataset["test"]),
        "output_path": "data/processed/dolly_processed"
    }
    
    print(f"\nData preparation complete:")
    print(f"  Train samples: {results['train_samples']:,}")
    print(f"  Validation samples: {results['val_samples']:,}")
    print(f"  Test samples: {results['test_samples']:,}")
    
    return results


# ============================================================================
# STEP 2: FINE-TUNING
# ============================================================================

def run_training_transformers(config: dict, output_dir: Path) -> Dict[str, Any]:
    """Fine-tune using Transformers + PEFT (QLoRA)."""
    print_step(2, "Fine-tuning with Transformers + PEFT (QLoRA)")
    
    from fine_tuning.train_transformers_peft import TransformersFinetuner
    
    finetuner = TransformersFinetuner(config=config)
    stats = finetuner.run_full_pipeline(
        data_path="data/processed/dolly_processed",
        merge_model=True
    )
    
    return {
        "framework": "transformers_peft",
        "training_time_minutes": stats.get("training_time_minutes", 0),
        "peak_memory_gb": stats.get("peak_memory_gb", 0),
        "train_loss": stats.get("train_loss", 0),
        "eval_loss": stats.get("eval_loss", 0),
        "output_dir": str(finetuner.output_dir)
    }


def run_training_unsloth(config: dict, output_dir: Path) -> Dict[str, Any]:
    """Fine-tune using Unsloth."""
    print_step(2, "Fine-tuning with Unsloth")
    
    from fine_tuning.train_unsloth import UnslothFinetuner
    
    finetuner = UnslothFinetuner(config=config)
    stats = finetuner.run_full_pipeline(
        data_path="data/processed/dolly_processed",
        merge_model=True
    )
    
    return {
        "framework": "unsloth",
        "training_time_minutes": stats.get("training_time_minutes", 0),
        "peak_memory_gb": stats.get("peak_memory_gb", 0),
        "train_loss": stats.get("train_loss", 0),
        "eval_loss": stats.get("eval_loss", 0),
        "output_dir": str(finetuner.output_dir)
    }


# ============================================================================
# STEP 3: EVALUATION
# ============================================================================

def run_evaluation(
    config: dict,
    model_path: str,
    model_name: str,
    output_dir: Path,
    num_samples: int = 50
) -> Dict[str, Any]:
    """Evaluate model quality using automatic metrics."""
    print_step(3, f"Evaluating {model_name}")
    
    from evaluation.evaluate_model import evaluate_model
    
    results = evaluate_model(
        model_path=model_path,
        model_name=model_name,
        num_samples=num_samples,
        output_dir=str(output_dir / "evaluation")
    )
    
    metrics = results.get("metrics", {})
    print(f"\nEvaluation Results:")
    print(f"  BLEU: {metrics.get('bleu', 0):.2f}")
    print(f"  ROUGE-1: {metrics.get('rouge_1', 0):.4f}")
    print(f"  ROUGE-2: {metrics.get('rouge_2', 0):.4f}")
    print(f"  ROUGE-L: {metrics.get('rouge_l', 0):.4f}")
    print(f"  BERTScore F1: {metrics.get('bertscore_f1', 0):.4f}")
    
    return results


# ============================================================================
# STEP 4: INFERENCE BENCHMARKING
# ============================================================================

def run_benchmark_single(
    config: dict,
    model_path: str,
    framework: str,
    output_dir: Path,
    num_prompts: int = 5,
    num_trials: int = 3
) -> Dict[str, Any]:
    """Benchmark inference performance for a single framework."""
    
    from evaluation.benchmark import InferenceBenchmarker
    
    benchmarker = InferenceBenchmarker(config=config)
    
    # Load test prompts
    test_prompts = load_test_prompts(num_prompts)
    
    # Run benchmark (repeat prompts for num_trials)
    all_prompts = test_prompts * num_trials
    
    benchmark_result = benchmarker.benchmark_framework(
        framework=framework,
        model_path=model_path,
        prompts=all_prompts
    )
    
    if benchmark_result is None:
        raise RuntimeError(f"Benchmark failed for {framework}")
    
    # Convert to dict format matching existing results
    results = {
        "framework": framework,
        "timestamp": benchmark_result.timestamp,
        "model_path": model_path,
        "config": {
            "num_prompts": num_prompts,
            "num_trials": num_trials,
            "max_new_tokens": benchmark_result.max_new_tokens
        },
        "results": {
            "avg_latency_s": benchmark_result.avg_latency_seconds,
            "std_latency_s": (benchmark_result.p90_latency_seconds - benchmark_result.avg_latency_seconds) / 1.28,  # Approximate std
            "avg_tokens_generated": benchmark_result.total_tokens_generated / len(all_prompts),
            "avg_throughput_tokens_per_s": benchmark_result.avg_throughput_tokens_per_sec,
            "std_throughput": 0.0,  # Not available from current benchmark
            "peak_memory_gb": benchmark_result.peak_memory_gb
        }
    }
    
    # Save results
    benchmark_dir = output_dir / "benchmarks"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = benchmark_dir / f"{framework}_benchmark.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    return results


def run_benchmark_all(
    config: dict,
    model_path: str,
    frameworks: List[str],
    output_dir: Path,
    num_prompts: int = 5,
    num_trials: int = 3
) -> Dict[str, Any]:
    """Benchmark all specified frameworks."""
    print_step(4, "Inference Benchmarking")
    
    all_results = {}
    
    for fw in frameworks:
        print(f"\nBenchmarking {fw}...")
        try:
            results = run_benchmark_single(
                config, model_path, fw, output_dir, num_prompts, num_trials
            )
            all_results[fw] = results
            
            res = results.get("results", {})
            print(f"  Latency: {res.get('avg_latency_s', 0):.3f}s")
            print(f"  Throughput: {res.get('avg_throughput_tokens_per_s', 0):.1f} tok/s")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            all_results[fw] = {"error": str(e)}
    
    return all_results


def load_test_prompts(num_prompts: int = 5) -> List[str]:
    """Load test prompts for benchmarking."""
    eval_file = Path("data/processed/eval_dataset/test_eval.json")
    
    if eval_file.exists():
        with open(eval_file) as f:
            data = [json.loads(line) for line in f]
        return [d["prompt"] for d in data[:num_prompts]]
    else:
        # Fallback prompts using TinyLlama format
        return [
            "<|system|>\nYou are a helpful assistant.</s>\n<|user|>\nWhat is machine learning?</s>\n<|assistant|>\n",
            "<|system|>\nYou are a helpful assistant.</s>\n<|user|>\nExplain neural networks.</s>\n<|assistant|>\n",
            "<|system|>\nYou are a helpful assistant.</s>\n<|user|>\nWhat is Python?</s>\n<|assistant|>\n",
            "<|system|>\nYou are a helpful assistant.</s>\n<|user|>\nHow does fine-tuning work?</s>\n<|assistant|>\n",
            "<|system|>\nYou are a helpful assistant.</s>\n<|user|>\nWhat are transformers?</s>\n<|assistant|>\n",
        ][:num_prompts]


# ============================================================================
# STEP 5: VISUALIZATIONS
# ============================================================================

def run_visualizations(output_dir: Path) -> Dict[str, Any]:
    """Generate all visualizations."""
    print_step(5, "Generating Visualizations")
    
    import subprocess
    result = subprocess.run(
        [sys.executable, "generate_visualizations.py"],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT
    )
    
    if result.returncode == 0:
        print("Visualizations generated successfully!")
        # Print output but limit lines
        for line in result.stdout.split('\n')[:20]:
            if line.strip():
                print(f"  {line}")
    else:
        print(f"Warning: {result.stderr}")
    
    return {"status": "completed", "output_dir": "results/plots"}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def find_latest_model(trainer: str) -> Optional[str]:
    """Find the latest trained model for the given trainer."""
    if trainer == "transformers":
        base_dir = Path("outputs/transformers")
    else:
        base_dir = Path("outputs/unsloth")
    
    if not base_dir.exists():
        return None
    
    # Find subdirectories sorted by name (timestamp)
    subdirs = sorted([d for d in base_dir.iterdir() if d.is_dir()], reverse=True)
    
    for subdir in subdirs:
        merged_path = subdir / "merged_model"
        if merged_path.exists():
            return str(merged_path)
        adapter_path = subdir / "adapter"
        if adapter_path.exists():
            return str(adapter_path)
    
    return None


def save_results(results: Dict[str, Any], output_dir: Path):
    """Save pipeline results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"pipeline_results_{timestamp}.json"
    
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {results_file}")
    return results_file


# ============================================================================
# MAIN PIPELINE RUNNER
# ============================================================================

def run_pipeline(args) -> Dict[str, Any]:
    """Run the pipeline based on the specified mode."""
    
    print_header("NLP II Final Project - Fine-tune Once, Serve Anywhere")
    print(f"Mode: {args.mode}")
    print(f"Trainer: {args.trainer}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print_gpu_info()
    
    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "mode": args.mode,
        "trainer": args.trainer,
        "config_path": args.config,
        "steps": {}
    }
    
    start_time = time.time()
    
    try:
        # ====================================================================
        # DATA PREPARATION
        # ====================================================================
        if args.mode in ["full", "data"]:
            results["steps"]["data"] = run_data_preparation(config, output_dir)
        
        # ====================================================================
        # TRAINING
        # ====================================================================
        if args.mode in ["full", "train"]:
            if args.mode == "full":
                # In full mode, train BOTH frameworks
                print("\n" + "="*60)
                print("Training with BOTH frameworks (Transformers + Unsloth)")
                print("="*60)
                
                # Train with Transformers first
                try:
                    results["steps"]["training_transformers"] = run_training_transformers(config, output_dir)
                    # Clear GPU memory
                    torch.cuda.empty_cache()
                except Exception as e:
                    print(f"WARNING: Transformers training failed: {e}")
                    results["steps"]["training_transformers"] = {"error": str(e)}
                
                # Then train with Unsloth
                try:
                    results["steps"]["training_unsloth"] = run_training_unsloth(config, output_dir)
                    # Clear GPU memory
                    torch.cuda.empty_cache()
                except Exception as e:
                    print(f"WARNING: Unsloth training failed: {e}")
                    results["steps"]["training_unsloth"] = {"error": str(e)}
            else:
                # In train mode, use the specified trainer
                if args.trainer == "transformers":
                    results["steps"]["training"] = run_training_transformers(config, output_dir)
                else:
                    results["steps"]["training"] = run_training_unsloth(config, output_dir)
        
        # ====================================================================
        # EVALUATION
        # ====================================================================
        if args.mode in ["full", "evaluate"]:
            if args.mode == "full":
                # In full mode, evaluate BOTH models
                for trainer in ["transformers", "unsloth"]:
                    model_path = find_latest_model(trainer)
                    if model_path:
                        print(f"\nUsing model: {model_path}")
                        try:
                            results["steps"][f"evaluation_{trainer}"] = run_evaluation(
                                config, model_path, trainer.capitalize(),
                                output_dir, args.eval_samples
                            )
                            torch.cuda.empty_cache()
                        except Exception as e:
                            print(f"WARNING: {trainer} evaluation failed: {e}")
                            results["steps"][f"evaluation_{trainer}"] = {"error": str(e)}
                    else:
                        print(f"WARNING: No {trainer} model found for evaluation.")
            else:
                model_path = find_latest_model(args.trainer)
                if model_path:
                    print(f"Using model: {model_path}")
                    results["steps"]["evaluation"] = run_evaluation(
                        config, model_path, args.trainer.capitalize(),
                        output_dir, args.eval_samples
                    )
                else:
                    print("WARNING: No trained model found for evaluation.")
                    print("Run training first with: python main.py --mode train")
        
        # ====================================================================
        # BENCHMARKING
        # ====================================================================
        if args.mode in ["full", "benchmark"]:
            # Use transformers model for benchmarking (most stable)
            model_path = find_latest_model("transformers")
            if model_path:
                print(f"Using model for benchmarks: {model_path}")
                try:
                    results["steps"]["benchmark"] = run_benchmark_all(
                        config, model_path, args.frameworks,
                        output_dir, args.benchmark_prompts, args.benchmark_trials
                    )
                    torch.cuda.empty_cache()
                except Exception as e:
                    print(f"WARNING: Benchmarking failed: {e}")
                    results["steps"]["benchmark"] = {"error": str(e)}
            else:
                print("WARNING: No trained model found for benchmarking.")
                print("Run training first with: python main.py --mode train")
        
        # ====================================================================
        # VISUALIZATIONS
        # ====================================================================
        if args.mode in ["full", "visualize"]:
            results["steps"]["visualizations"] = run_visualizations(output_dir)
        
        # ====================================================================
        # SAVE RESULTS
        # ====================================================================
        total_time = time.time() - start_time
        results["total_time_seconds"] = total_time
        results["total_time_minutes"] = total_time / 60
        results["status"] = "completed"
        
        save_results(results, output_dir)
        
        print_header("Pipeline Complete!")
        print(f"Total time: {total_time/60:.1f} minutes ({total_time:.0f} seconds)")
        print(f"Output directory: {output_dir}")
        
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user.")
        results["status"] = "interrupted"
        save_results(results, output_dir)
        
    except Exception as e:
        print(f"\nERROR: Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        results["status"] = "failed"
        results["error"] = str(e)
        save_results(results, output_dir)
    
    return results


# ============================================================================
# CLI INTERFACE
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="NLP II Final Project - Fine-tune Once, Serve Anywhere",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Full pipeline:
    python main.py --mode full
    
  Individual steps:
    python main.py --mode data                          # Prepare dataset
    python main.py --mode train --trainer transformers  # Train with Transformers+PEFT
    python main.py --mode train --trainer unsloth       # Train with Unsloth
    python main.py --mode evaluate                      # Evaluate model
    python main.py --mode benchmark                     # Benchmark all frameworks
    python main.py --mode visualize                     # Generate plots
    
  Custom configuration:
    python main.py --mode train --trainer transformers --config custom_config.yaml
    python main.py --mode benchmark --frameworks transformers vllm --benchmark-trials 5
        """
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["full", "data", "train", "evaluate", "benchmark", "visualize"],
        default="full",
        help="Pipeline mode (default: full)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file (default: configs/config.yaml)"
    )
    
    parser.add_argument(
        "--trainer",
        type=str,
        choices=["transformers", "unsloth"],
        default="transformers",
        help="Training framework (default: transformers)"
    )
    
    parser.add_argument(
        "--frameworks",
        type=str,
        nargs="+",
        default=["transformers", "unsloth", "vllm", "ollama"],
        help="Inference frameworks to benchmark (default: all four)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory (default: results)"
    )
    
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=50,
        help="Number of samples for evaluation (default: 50)"
    )
    
    parser.add_argument(
        "--benchmark-prompts",
        type=int,
        default=5,
        help="Number of prompts for benchmarking (default: 5)"
    )
    
    parser.add_argument(
        "--benchmark-trials",
        type=int,
        default=3,
        help="Number of trials per benchmark (default: 3)"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Check CUDA for training modes
    if args.mode in ["full", "train"]:
        if not torch.cuda.is_available():
            print("ERROR: CUDA not available. Training requires a GPU.")
            print("For CPU-only operations, use: --mode data or --mode visualize")
            sys.exit(1)
    
    # Run pipeline
    run_pipeline(args)


if __name__ == "__main__":
    main()
