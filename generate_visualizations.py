#!/usr/bin/env python3
"""
Generate All Visualizations for NLP II Final Project Report
============================================================
Creates publication-ready figures for:
1. Training comparison (Transformers vs Unsloth)
2. Inference benchmark (4 frameworks)
3. Quality metrics (BLEU, ROUGE, BERTScore)
4. Human evaluation results
5. Before/After fine-tuning comparison
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns

# Configuration
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['figure.dpi'] = 150

OUTPUT_DIR = Path("results/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Color palette
COLORS = {
    "transformers": "#E74C3C",  # Red
    "unsloth": "#27AE60",       # Green
    "vllm": "#3498DB",          # Blue
    "ollama": "#9B59B6",        # Purple
    "base": "#95A5A6",          # Gray
}


def load_all_data():
    """Load all result files."""
    data = {}
    
    # Benchmarks
    for fw in ["transformers", "unsloth", "vllm", "ollama"]:
        path = Path(f"results/benchmarks/{fw}_benchmark.json")
        if path.exists():
            with open(path) as f:
                data[f"{fw}_benchmark"] = json.load(f)
    
    # Evaluations (including base model)
    for name in ["transformers_eval", "unsloth_eval", "base_model_eval", "before_after_comparison", "human_eval_completed"]:
        path = Path(f"results/evaluation/{name}.json")
        if path.exists():
            with open(path) as f:
                data[name] = json.load(f)
    
    # Training stats
    for fw in ["transformers", "unsloth"]:
        # Find latest training output
        output_dir = Path(f"outputs/{fw}")
        if output_dir.exists():
            subdirs = sorted([d for d in output_dir.iterdir() if d.is_dir()])
            if subdirs:
                stats_path = subdirs[-1] / "training_stats.json"
                if stats_path.exists():
                    with open(stats_path) as f:
                        data[f"{fw}_training"] = json.load(f)
    
    return data


def plot_inference_benchmark(data):
    """
    Plot 1: Inference Benchmark Comparison (4 frameworks)
    - Latency
    - Throughput  
    - Memory
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    frameworks = ["transformers", "unsloth", "vllm", "ollama"]
    labels = ["Transformers", "Unsloth", "vLLM", "Ollama"]
    colors = [COLORS[fw] for fw in frameworks]
    
    # Extract data
    latencies = []
    latency_stds = []
    throughputs = []
    throughput_stds = []
    memories = []
    
    for fw in frameworks:
        bm = data.get(f"{fw}_benchmark", {}).get("results", {})
        latencies.append(bm.get("avg_latency_s", 0))
        latency_stds.append(bm.get("std_latency_s", 0))
        throughputs.append(bm.get("avg_throughput_tokens_per_s", 0))
        throughput_stds.append(bm.get("std_throughput", 0))
        memories.append(bm.get("peak_memory_gb", 0))
    
    # Plot 1: Latency (lower is better)
    ax1 = axes[0]
    bars1 = ax1.bar(labels, latencies, yerr=latency_stds, capsize=5, 
                    color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel("Latency (seconds)")
    ax1.set_title("Average Latency\n(lower is better)", fontweight="bold")
    ax1.set_ylim(0, max(latencies) * 1.3)
    for bar, val in zip(bars1, latencies):
        ax1.annotate(f'{val:.3f}s', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax1.tick_params(axis='x', rotation=15)
    
    # Plot 2: Throughput (higher is better)
    ax2 = axes[1]
    bars2 = ax2.bar(labels, throughputs, yerr=throughput_stds, capsize=5,
                    color=colors, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel("Tokens per second")
    ax2.set_title("Throughput\n(higher is better)", fontweight="bold")
    ax2.set_ylim(0, max(throughputs) * 1.2)
    for bar, val in zip(bars2, throughputs):
        ax2.annotate(f'{val:.0f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax2.tick_params(axis='x', rotation=15)
    
    # Plot 3: Memory (lower is better)
    ax3 = axes[2]
    # Note: vLLM reports 0 for memory in benchmark, handle this
    mem_display = [m if m > 0 else np.nan for m in memories]
    bars3 = ax3.bar(labels, mem_display, color=colors, edgecolor='black', linewidth=1.5)
    ax3.axhline(y=12, color='red', linestyle='--', linewidth=2, label='GPU Limit (12GB)')
    ax3.set_ylabel("Memory (GB)")
    ax3.set_title("Peak GPU Memory\n(lower is better)", fontweight="bold")
    ax3.legend(loc='upper right')
    ax3.set_ylim(0, 14)
    for bar, val in zip(bars3, memories):
        if val > 0:
            ax3.annotate(f'{val:.2f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
        else:
            ax3.annotate('N/A', xy=(bar.get_x() + bar.get_width()/2, 0.5),
                        ha='center', va='bottom', fontsize=10, fontweight='bold', color='gray')
    ax3.tick_params(axis='x', rotation=15)
    
    plt.suptitle("Inference Benchmark: Framework Comparison", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_inference_benchmark.png", bbox_inches='tight', dpi=150)
    plt.savefig(OUTPUT_DIR / "01_inference_benchmark.pdf", bbox_inches='tight')
    print("Saved: 01_inference_benchmark.png/pdf")
    plt.close()


def plot_training_comparison(data):
    """
    Plot 2: Training Efficiency Comparison (Transformers vs Unsloth)
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    frameworks = ["transformers", "unsloth"]
    labels = ["Transformers\n+ PEFT", "Unsloth"]
    colors = [COLORS["transformers"], COLORS["unsloth"]]
    
    # Extract training stats
    times = []
    memories = []
    losses = []
    
    for fw in frameworks:
        stats = data.get(f"{fw}_training", {})
        times.append(stats.get("training_time_minutes", 0))
        memories.append(stats.get("peak_memory_gb", 0))
        losses.append(stats.get("train_loss", 0))
    
    # Plot 1: Training Time
    ax1 = axes[0]
    bars1 = ax1.bar(labels, times, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel("Minutes")
    ax1.set_title("Training Time\n(lower is better)", fontweight="bold")
    for bar, val in zip(bars1, times):
        ax1.annotate(f'{val:.1f} min', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Plot 2: Peak Memory
    ax2 = axes[1]
    bars2 = ax2.bar(labels, memories, color=colors, edgecolor='black', linewidth=1.5)
    ax2.axhline(y=12, color='red', linestyle='--', linewidth=2, label='GPU Limit (12GB)')
    ax2.set_ylabel("GPU Memory (GB)")
    ax2.set_title("Peak Memory Usage\n(lower is better)", fontweight="bold")
    ax2.legend(loc='upper right')
    ax2.set_ylim(0, 14)
    for bar, val in zip(bars2, memories):
        ax2.annotate(f'{val:.2f} GB', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Plot 3: Final Training Loss
    ax3 = axes[2]
    bars3 = ax3.bar(labels, losses, color=colors, edgecolor='black', linewidth=1.5)
    ax3.set_ylabel("Loss")
    ax3.set_title("Final Training Loss\n(lower is better)", fontweight="bold")
    for bar, val in zip(bars3, losses):
        ax3.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.suptitle("Training Efficiency: Transformers+PEFT vs Unsloth", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_training_comparison.png", bbox_inches='tight', dpi=150)
    plt.savefig(OUTPUT_DIR / "02_training_comparison.pdf", bbox_inches='tight')
    print("Saved: 02_training_comparison.png/pdf")
    plt.close()


def plot_quality_metrics(data):
    """
    Plot 3: Quality Metrics Comparison (BLEU, ROUGE, BERTScore)
    Comparing Base Model vs Transformers vs Unsloth fine-tuned models
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Get metrics
    base_metrics = data.get("base_model_eval", {}).get("metrics", {})
    trans_metrics = data.get("transformers_eval", {}).get("metrics", {})
    unsloth_metrics = data.get("unsloth_eval", {}).get("metrics", {})
    
    width = 0.25
    
    # Plot 1: BLEU scores
    ax1 = axes[0]
    bleu_labels = ["BLEU", "BLEU-1", "BLEU-2"]
    base_bleu = [
        base_metrics.get("bleu", 0),
        base_metrics.get("bleu_1", 0),
        base_metrics.get("bleu_2", 0)
    ]
    trans_bleu = [
        trans_metrics.get("bleu", 0),
        trans_metrics.get("bleu_1", 0),
        trans_metrics.get("bleu_2", 0)
    ]
    unsloth_bleu = [
        unsloth_metrics.get("bleu", 0),
        unsloth_metrics.get("bleu_1", 0),
        unsloth_metrics.get("bleu_2", 0)
    ]
    
    x1 = np.arange(len(bleu_labels))
    bars0 = ax1.bar(x1 - width, base_bleu, width, label='Base Model', 
                    color=COLORS["base"], edgecolor='black')
    bars1 = ax1.bar(x1, trans_bleu, width, label='Transformers', 
                    color=COLORS["transformers"], edgecolor='black')
    bars2 = ax1.bar(x1 + width, unsloth_bleu, width, label='Unsloth',
                    color=COLORS["unsloth"], edgecolor='black')
    
    ax1.set_ylabel("Score")
    ax1.set_title("BLEU Scores", fontweight="bold")
    ax1.set_xticks(x1)
    ax1.set_xticklabels(bleu_labels)
    ax1.legend(loc='upper right', fontsize=8)
    ax1.set_ylim(0, max(max(base_bleu), max(trans_bleu), max(unsloth_bleu)) * 1.25)
    
    for bar in bars0:
        ax1.annotate(f'{bar.get_height():.1f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=8)
    for bar in bars1:
        ax1.annotate(f'{bar.get_height():.1f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        ax1.annotate(f'{bar.get_height():.1f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=8)
    
    # Plot 2: ROUGE scores
    ax2 = axes[1]
    rouge_labels = ["ROUGE-1", "ROUGE-2", "ROUGE-L"]
    base_rouge = [
        base_metrics.get("rouge_1", 0),
        base_metrics.get("rouge_2", 0),
        base_metrics.get("rouge_l", 0)
    ]
    trans_rouge = [
        trans_metrics.get("rouge_1", 0),
        trans_metrics.get("rouge_2", 0),
        trans_metrics.get("rouge_l", 0)
    ]
    unsloth_rouge = [
        unsloth_metrics.get("rouge_1", 0),
        unsloth_metrics.get("rouge_2", 0),
        unsloth_metrics.get("rouge_l", 0)
    ]
    
    x2 = np.arange(len(rouge_labels))
    bars3 = ax2.bar(x2 - width, base_rouge, width, label='Base Model',
                    color=COLORS["base"], edgecolor='black')
    bars4 = ax2.bar(x2, trans_rouge, width, label='Transformers',
                    color=COLORS["transformers"], edgecolor='black')
    bars5 = ax2.bar(x2 + width, unsloth_rouge, width, label='Unsloth',
                    color=COLORS["unsloth"], edgecolor='black')
    
    ax2.set_ylabel("F1 Score")
    ax2.set_title("ROUGE Scores", fontweight="bold")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(rouge_labels)
    ax2.legend(loc='upper right', fontsize=8)
    ax2.set_ylim(0, 0.5)
    
    for bar in bars3:
        ax2.annotate(f'{bar.get_height():.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=8)
    for bar in bars4:
        ax2.annotate(f'{bar.get_height():.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=8)
    for bar in bars5:
        ax2.annotate(f'{bar.get_height():.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=8)
    
    # Plot 3: BERTScore
    ax3 = axes[2]
    bert_labels = ["Precision", "Recall", "F1"]
    base_bert = [
        base_metrics.get("bertscore_precision", 0),
        base_metrics.get("bertscore_recall", 0),
        base_metrics.get("bertscore_f1", 0)
    ]
    trans_bert = [
        trans_metrics.get("bertscore_precision", 0),
        trans_metrics.get("bertscore_recall", 0),
        trans_metrics.get("bertscore_f1", 0)
    ]
    unsloth_bert = [
        unsloth_metrics.get("bertscore_precision", 0),
        unsloth_metrics.get("bertscore_recall", 0),
        unsloth_metrics.get("bertscore_f1", 0)
    ]
    
    x3 = np.arange(len(bert_labels))
    bars6 = ax3.bar(x3 - width, base_bert, width, label='Base Model',
                    color=COLORS["base"], edgecolor='black')
    bars7 = ax3.bar(x3, trans_bert, width, label='Transformers',
                    color=COLORS["transformers"], edgecolor='black')
    bars8 = ax3.bar(x3 + width, unsloth_bert, width, label='Unsloth',
                    color=COLORS["unsloth"], edgecolor='black')
    
    ax3.set_ylabel("Score")
    ax3.set_title("BERTScore (Semantic Similarity)", fontweight="bold")
    ax3.set_xticks(x3)
    ax3.set_xticklabels(bert_labels)
    ax3.legend(loc='upper right', fontsize=8)
    ax3.set_ylim(0.7, 1.0)  # BERTScore typically ranges 0.7-1.0
    
    for bar in bars6:
        ax3.annotate(f'{bar.get_height():.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=8)
    for bar in bars7:
        ax3.annotate(f'{bar.get_height():.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=8)
    for bar in bars8:
        ax3.annotate(f'{bar.get_height():.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=8)
    
    plt.suptitle("Automatic Evaluation: BLEU, ROUGE & BERTScore", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "03_quality_metrics.png", bbox_inches='tight', dpi=150)
    plt.savefig(OUTPUT_DIR / "03_quality_metrics.pdf", bbox_inches='tight')
    print("Saved: 03_quality_metrics.png/pdf")
    plt.close()


def plot_human_evaluation(data):
    """
    Plot 4: Human Evaluation Results (5 prompts, 3 criteria)
    """
    human_eval = data.get("human_eval_completed", [])
    if not human_eval:
        print("Warning: No human evaluation data found")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    criteria = ["helpfulness", "factuality", "instruction_following"]
    titles = ["Helpfulness", "Factuality", "Instruction Following"]
    
    # Extract scores
    trans_scores = {c: [] for c in criteria}
    unsloth_scores = {c: [] for c in criteria}
    
    for sample in human_eval:
        for c in criteria:
            if "eval_transformers" in sample:
                trans_scores[c].append(sample["eval_transformers"].get(c, 0))
            if "eval_unsloth" in sample:
                unsloth_scores[c].append(sample["eval_unsloth"].get(c, 0))
    
    x = np.arange(len(human_eval))
    width = 0.35
    
    for ax, criterion, title in zip(axes, criteria, titles):
        trans = trans_scores[criterion]
        unsloth = unsloth_scores[criterion]
        
        bars1 = ax.bar(x - width/2, trans, width, label='Transformers',
                      color=COLORS["transformers"], edgecolor='black', alpha=0.8)
        bars2 = ax.bar(x + width/2, unsloth, width, label='Unsloth',
                      color=COLORS["unsloth"], edgecolor='black', alpha=0.8)
        
        ax.set_ylabel("Score (1-5)")
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"P{i+1}" for i in range(len(human_eval))])
        ax.set_ylim(0, 5.5)
        ax.axhline(y=np.mean(trans), color=COLORS["transformers"], linestyle='--', 
                  linewidth=2, alpha=0.7)
        ax.axhline(y=np.mean(unsloth), color=COLORS["unsloth"], linestyle='--',
                  linewidth=2, alpha=0.7)
        ax.legend(loc='upper right')
        
        # Add mean annotations
        ax.annotate(f'Mean: {np.mean(trans):.1f}', xy=(0.02, 0.95), xycoords='axes fraction',
                   color=COLORS["transformers"], fontsize=9, fontweight='bold')
        ax.annotate(f'Mean: {np.mean(unsloth):.1f}', xy=(0.02, 0.88), xycoords='axes fraction',
                   color=COLORS["unsloth"], fontsize=9, fontweight='bold')
    
    plt.suptitle("Human Evaluation: 5 Test Prompts (Scale 1-5)", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "04_human_evaluation.png", bbox_inches='tight', dpi=150)
    plt.savefig(OUTPUT_DIR / "04_human_evaluation.pdf", bbox_inches='tight')
    print("Saved: 04_human_evaluation.png/pdf")
    plt.close()


def plot_before_after(data):
    """
    Plot 5: Before/After Fine-tuning Comparison (BLEU, ROUGE-L, BERTScore F1)
    """
    comparison = data.get("before_after_comparison", {})
    if not comparison:
        print("Warning: No before/after comparison data found")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    models = ["Base Model", "Transformers\nFine-tuned", "Unsloth\nFine-tuned"]
    colors = [COLORS["base"], COLORS["transformers"], COLORS["unsloth"]]
    
    # BLEU scores
    bleu_scores = [
        comparison.get("base_model", {}).get("bleu", 0),
        comparison.get("finetuned_transformers", {}).get("bleu", 0),
        comparison.get("finetuned_unsloth", {}).get("bleu", 0)
    ]
    
    # ROUGE-L scores
    rouge_scores = [
        comparison.get("base_model", {}).get("rouge_l", 0),
        comparison.get("finetuned_transformers", {}).get("rouge_l", 0),
        comparison.get("finetuned_unsloth", {}).get("rouge_l", 0)
    ]
    
    # BERTScore F1
    bertscore_scores = [
        comparison.get("base_model", {}).get("bertscore_f1", 0),
        comparison.get("finetuned_transformers", {}).get("bertscore_f1", 0),
        comparison.get("finetuned_unsloth", {}).get("bertscore_f1", 0)
    ]
    
    # Plot BLEU
    ax1 = axes[0]
    bars1 = ax1.bar(models, bleu_scores, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel("BLEU Score")
    ax1.set_title("BLEU Score Comparison", fontweight="bold")
    ax1.set_ylim(0, max(bleu_scores) * 1.3)
    for bar, val in zip(bars1, bleu_scores):
        ax1.annotate(f'{val:.2f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax1.tick_params(axis='x', rotation=10)
    
    # Add improvement arrows/annotations
    if bleu_scores[1] > bleu_scores[0]:
        improvement = ((bleu_scores[1] - bleu_scores[0]) / bleu_scores[0]) * 100
        ax1.annotate(f'+{improvement:.1f}%', xy=(1, bleu_scores[1]*1.1),
                    ha='center', fontsize=9, color='green', fontweight='bold')
    
    # Plot ROUGE-L
    ax2 = axes[1]
    bars2 = ax2.bar(models, rouge_scores, color=colors, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel("ROUGE-L Score")
    ax2.set_title("ROUGE-L Score Comparison", fontweight="bold")
    ax2.set_ylim(0, max(rouge_scores) * 1.3)
    for bar, val in zip(bars2, rouge_scores):
        ax2.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax2.tick_params(axis='x', rotation=10)
    
    if rouge_scores[1] > rouge_scores[0]:
        improvement = ((rouge_scores[1] - rouge_scores[0]) / rouge_scores[0]) * 100
        ax2.annotate(f'+{improvement:.1f}%', xy=(1, rouge_scores[1]*1.1),
                    ha='center', fontsize=9, color='green', fontweight='bold')
    
    # Plot BERTScore F1
    ax3 = axes[2]
    bars3 = ax3.bar(models, bertscore_scores, color=colors, edgecolor='black', linewidth=1.5)
    ax3.set_ylabel("BERTScore F1")
    ax3.set_title("BERTScore F1 Comparison", fontweight="bold")
    ax3.set_ylim(0.75, max(bertscore_scores) * 1.05)  # Start from 0.75 for better visualization
    for bar, val in zip(bars3, bertscore_scores):
        ax3.annotate(f'{val:.4f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax3.tick_params(axis='x', rotation=10)
    
    if bertscore_scores[1] > bertscore_scores[0] and bertscore_scores[0] > 0:
        improvement = ((bertscore_scores[1] - bertscore_scores[0]) / bertscore_scores[0]) * 100
        ax3.annotate(f'+{improvement:.1f}%', xy=(1, bertscore_scores[1]*1.01),
                    ha='center', fontsize=9, color='green', fontweight='bold')
    
    plt.suptitle("Effect of Fine-tuning: Base Model vs Fine-tuned", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "05_before_after.png", bbox_inches='tight', dpi=150)
    plt.savefig(OUTPUT_DIR / "05_before_after.pdf", bbox_inches='tight')
    print("Saved: 05_before_after.png/pdf")
    plt.close()


def plot_summary_table(data):
    """
    Plot 6: Summary Table Figure
    """
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.axis('off')
    
    # Prepare data for table
    columns = ["Metric", "Base Model", "Transformers+PEFT", "Unsloth", "vLLM", "Ollama"]
    
    # Training metrics
    trans_train = data.get("transformers_training", {})
    unsloth_train = data.get("unsloth_training", {})
    
    # Benchmark metrics
    trans_bm = data.get("transformers_benchmark", {}).get("results", {})
    unsloth_bm = data.get("unsloth_benchmark", {}).get("results", {})
    vllm_bm = data.get("vllm_benchmark", {}).get("results", {})
    ollama_bm = data.get("ollama_benchmark", {}).get("results", {})
    
    # Quality metrics
    base_eval = data.get("base_model_eval", {}).get("metrics", {})
    trans_eval = data.get("transformers_eval", {}).get("metrics", {})
    unsloth_eval = data.get("unsloth_eval", {}).get("metrics", {})
    
    rows = [
        ["Training Time (min)", 
         "-",
         f"{trans_train.get('training_time_minutes', 'N/A'):.1f}" if trans_train else "N/A",
         f"{unsloth_train.get('training_time_minutes', 'N/A'):.1f}" if unsloth_train else "N/A",
         "-", "-"],
        ["Training Memory (GB)",
         "-",
         f"{trans_train.get('peak_memory_gb', 'N/A'):.2f}" if trans_train else "N/A",
         f"{unsloth_train.get('peak_memory_gb', 'N/A'):.2f}" if unsloth_train else "N/A",
         "-", "-"],
        ["Final Loss",
         "-",
         f"{trans_train.get('train_loss', 'N/A'):.3f}" if trans_train else "N/A",
         f"{unsloth_train.get('train_loss', 'N/A'):.3f}" if unsloth_train else "N/A",
         "-", "-"],
        ["", "", "", "", "", ""],  # Separator
        ["Inference Latency (s)",
         "-",
         f"{trans_bm.get('avg_latency_s', 0):.3f}",
         f"{unsloth_bm.get('avg_latency_s', 0):.3f}",
         f"{vllm_bm.get('avg_latency_s', 0):.3f}",
         f"{ollama_bm.get('avg_latency_s', 0):.3f}"],
        ["Throughput (tok/s)",
         "-",
         f"{trans_bm.get('avg_throughput_tokens_per_s', 0):.1f}",
         f"{unsloth_bm.get('avg_throughput_tokens_per_s', 0):.1f}",
         f"{vllm_bm.get('avg_throughput_tokens_per_s', 0):.1f}",
         f"{ollama_bm.get('avg_throughput_tokens_per_s', 0):.1f}"],
        ["", "", "", "", "", ""],  # Separator
        ["BLEU Score",
         f"{base_eval.get('bleu', 0):.2f}",
         f"{trans_eval.get('bleu', 0):.2f}",
         f"{unsloth_eval.get('bleu', 0):.2f}",
         "-", "-"],
        ["ROUGE-L",
         f"{base_eval.get('rouge_l', 0):.3f}",
         f"{trans_eval.get('rouge_l', 0):.3f}",
         f"{unsloth_eval.get('rouge_l', 0):.3f}",
         "-", "-"],
        ["BERTScore F1",
         f"{base_eval.get('bertscore_f1', 0):.4f}",
         f"{trans_eval.get('bertscore_f1', 0):.4f}",
         f"{unsloth_eval.get('bertscore_f1', 0):.4f}",
         "-", "-"],
    ]
    
    table = ax.table(
        cellText=rows,
        colLabels=columns,
        loc='center',
        cellLoc='center',
        colColours=['#3498DB'] * len(columns),
        colWidths=[0.20, 0.15, 0.17, 0.15, 0.15, 0.15]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)
    
    # Style header
    for j in range(len(columns)):
        table[(0, j)].set_text_props(weight='bold', color='white')
    
    # Style metric column
    for i in range(len(rows)):
        table[(i+1, 0)].set_text_props(weight='bold')
    
    plt.title("Complete Results Summary", fontsize=16, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "06_summary_table.png", bbox_inches='tight', dpi=150)
    plt.savefig(OUTPUT_DIR / "06_summary_table.pdf", bbox_inches='tight')
    print("Saved: 06_summary_table.png/pdf")
    plt.close()


def plot_radar_comparison(data):
    """
    Plot 7: Radar Chart comparing all 4 frameworks
    """
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    frameworks = ["transformers", "unsloth", "vllm", "ollama"]
    labels = ["Transformers", "Unsloth", "vLLM", "Ollama"]
    
    # Metrics to compare (normalized 0-1, higher is better)
    categories = ["Speed\n(1/Latency)", "Throughput", "Memory\nEfficiency", "Setup\nEase"]
    num_vars = len(categories)
    
    # Extract and normalize data
    latencies = []
    throughputs = []
    memories = []
    
    for fw in frameworks:
        bm = data.get(f"{fw}_benchmark", {}).get("results", {})
        latencies.append(bm.get("avg_latency_s", 1))
        throughputs.append(bm.get("avg_throughput_tokens_per_s", 0))
        mem = bm.get("peak_memory_gb", 2)
        memories.append(mem if mem > 0 else 2)  # Default for missing
    
    # Normalize (0-1 scale)
    def normalize(vals, invert=False):
        min_v, max_v = min(vals), max(vals)
        if max_v == min_v:
            return [0.5] * len(vals)
        if invert:
            return [(max_v - v) / (max_v - min_v) for v in vals]
        return [(v - min_v) / (max_v - min_v) for v in vals]
    
    speed_norm = normalize(latencies, invert=True)  # Lower latency = better
    throughput_norm = normalize(throughputs)  # Higher = better
    memory_norm = normalize(memories, invert=True)  # Lower memory = better
    
    # Setup ease (subjective, based on typical experience)
    setup_ease = [0.7, 0.8, 0.5, 0.9]  # Ollama easiest, vLLM hardest
    
    # Combine scores
    scores = {
        fw: [speed_norm[i], throughput_norm[i], memory_norm[i], setup_ease[i]]
        for i, fw in enumerate(frameworks)
    }
    
    # Compute angle for each axis
    angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
    angles += angles[:1]  # Close the loop
    
    # Plot each framework
    for fw, label in zip(frameworks, labels):
        values = scores[fw]
        values += values[:1]  # Close the loop
        
        ax.plot(angles, values, 'o-', linewidth=2, label=label, color=COLORS[fw])
        ax.fill(angles, values, alpha=0.25, color=COLORS[fw])
    
    # Set category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=11)
    ax.set_ylim(0, 1)
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    plt.title("Framework Comparison Radar Chart\n(normalized scores, higher is better)", 
              fontsize=14, fontweight="bold", y=1.08)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "07_radar_comparison.png", bbox_inches='tight', dpi=150)
    plt.savefig(OUTPUT_DIR / "07_radar_comparison.pdf", bbox_inches='tight')
    print("Saved: 07_radar_comparison.png/pdf")
    plt.close()


def main():
    """Generate all visualizations."""
    print("=" * 60)
    print("Generating Visualizations for NLP II Final Project")
    print("=" * 60)
    print()
    
    # Load all data
    print("Loading data...")
    data = load_all_data()
    print(f"Loaded {len(data)} data files")
    print()
    
    # Generate all plots
    print("Generating plots...")
    print("-" * 40)
    
    plot_inference_benchmark(data)
    plot_training_comparison(data)
    plot_quality_metrics(data)
    plot_human_evaluation(data)
    plot_before_after(data)
    plot_summary_table(data)
    plot_radar_comparison(data)
    
    print("-" * 40)
    print()
    print(f"All visualizations saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
