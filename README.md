# Fine-tune Once, Serve Anywhere

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red.svg)
![CUDA](https://img.shields.io/badge/CUDA-11.8+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**Comparing Fine-tuning and Inference Pipelines with LLMs**

*NLP II Final Project - 2024/2025*

</div>

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Hardware Requirements](#hardware-requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Framework Comparison](#framework-comparison)
- [Usage Guide](#usage-guide)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This project implements a comprehensive comparison of fine-tuning and inference pipelines for Large Language Models (LLMs). We fine-tune **TinyLlama-1.1B-Chat** on the **Databricks Dolly 15k** dataset using **QLoRA** and compare inference performance across four frameworks:

| Framework | Purpose | Key Features |
|-----------|---------|--------------|
| **Transformers + PEFT** | Fine-tuning & Inference | QLoRA, 4-bit quantization |
| **Unsloth** | Optimized Fine-tuning | 2-5x faster training |
| **vLLM** | High-throughput Inference | PagedAttention, continuous batching |
| **Ollama** | Local Deployment | Easy model serving, GGUF format |

### Key Features

- Memory-efficient fine-tuning with QLoRA (4-bit quantization)
- Comprehensive benchmarking across 4 inference frameworks
- Automatic evaluation with BLEU, ROUGE-L, and BERTScore
- Human evaluation interface for qualitative assessment
- GPU-optimized for NVIDIA RTX 4070 SUPER (12GB VRAM)

---

## Project Structure

```
Final-Project-NLPII/
├── README.md                       # This file
├── requirements.txt                # Python dependencies
├── project_specification.txt       # Original project requirements
│
├── configs/
│   └── config.yaml                # Central configuration file
│
├── data/
│   ├── prepare_data.py            # Dataset preparation & EDA
│   └── processed/                 # Cached processed data
│
├── fine_tuning/
│   ├── train_transformers_peft.py # Transformers + PEFT training
│   └── train_unsloth_simple.py    # Unsloth optimized training
│
├── inference/
│   └── inference_engines.py       # Unified inference interface
│
├── evaluation/
│   ├── evaluate_model.py          # BLEU, ROUGE, BERTScore
│   ├── complete_evaluation.py     # Complete evaluation pipeline
│   ├── benchmark_vllm.py          # vLLM inference benchmark
│   └── benchmark_ollama.py        # Ollama inference benchmark
│
├── results/
│   ├── benchmarks/                # Benchmark results
│   ├── evaluation/                # Evaluation metrics
│   └── checkpoints/               # Model checkpoints
│
└── notebooks/
    └── analysis.ipynb             # Exploratory analysis
```

---

## Hardware Requirements

### Minimum Requirements
- GPU: NVIDIA GPU with 8GB+ VRAM
- RAM: 16GB system memory
- Storage: 20GB free disk space
- CUDA: 11.8 or higher

### Recommended (Development Setup)
- GPU: NVIDIA RTX 4070 SUPER (12GB VRAM)
- RAM: 32GB system memory
- Storage: 50GB SSD
- CUDA: 12.1

### GPU Memory Usage Estimates

| Operation | Memory Usage |
|-----------|-------------|
| Fine-tuning (QLoRA, batch=4) | ~8-10GB |
| Inference (Transformers) | ~4-6GB |
| Inference (vLLM) | ~5-7GB |
| Inference (Ollama) | ~3-5GB |

---

## Installation

### Automated Setup (Recommended)

We provide an automated setup script that handles everything:

```bash
# Make script executable
chmod +x setup.sh

# Run setup (creates venv, installs dependencies)
bash setup.sh
```

The script will:
1. Create virtual environment (`venv_pipeline/`)
2. Install PyTorch with CUDA 12.1
3. Install all project dependencies
4. Optionally install Unsloth for optimization
5. Verify all installations

Then activate the environment:
```bash
source venv_pipeline/bin/activate
```

### Manual Installation

**Step 1: Clone the Repository**
```bash
cd Final-Project-NLPII
```

**Step 2: Create Virtual Environment**
```bash
# Using venv
python3 -m venv venv_pipeline
source venv_pipeline/bin/activate  # Linux/Mac
# or on Windows: venv_pipeline\Scripts\activate
```

**Step 3: Install PyTorch**
```bash
# For CUDA 12.1 (recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Step 4: Install Project Dependencies**
```bash
pip install -r requirements.txt
```

**Step 5: (Optional) Install Unsloth**
```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

**Step 6: (Optional) Install Ollama**
```bash
# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve
```

**Step 7: Verify Installation**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

---

## Quick Start

### Full Setup and Execution

```bash
# 1. Automated setup (one-time)
bash setup.sh

# 2. Activate environment
source venv_pipeline/bin/activate

# 3. Run entire pipeline
python main.py --mode full

# Or use the bash wrapper
./run_pipeline.sh full
```

### Option 1: Run Full Pipeline

```bash
# Everything: data prep, training, evaluation, benchmarks, visualizations
./run_pipeline.sh full

# Or using Python
python main.py --mode full
```

**Expected execution time**: ~2-3 hours (varies by GPU)

### Option 2: Individual Steps

```bash
# Step 1: Prepare dataset (~2-3 minutes)
python main.py --mode data

# Step 2: Fine-tune with Transformers + PEFT (~90 minutes)
python main.py --mode train --trainer transformers

# Step 2 (alt): Fine-tune with Unsloth (~45 minutes, faster)
python main.py --mode train --trainer unsloth

# Step 3: Evaluate both models (~30 minutes)
python main.py --mode evaluate

# Step 4: Benchmark inference frameworks (~20 minutes)
python main.py --mode benchmark

# Step 5: Generate visualizations (~10 seconds)
python main.py --mode visualize
```

### Option 3: Using Bash Script

```bash
# Individual steps via bash script
./run_pipeline.sh data                    # Only data preparation
./run_pipeline.sh train                   # Train both frameworks
./run_pipeline.sh train-transformers      # Transformers + PEFT only
./run_pipeline.sh train-unsloth           # Unsloth only
./run_pipeline.sh evaluate                # Evaluate models
./run_pipeline.sh benchmark               # Run benchmarks
./run_pipeline.sh visualize               # Generate plots
```

---

## Detailed Usage

### 1. Prepare Data

```bash
python main.py --mode data
```

This will:
- Download Databricks Dolly 15k dataset
- Split into train (12k), validation (2k), test (1k)
- Format prompts in TinyLlama ChatML template
- Save processed data to `data/processed/`

### 2. Fine-tune Model

**Option A: Using Transformers + PEFT (QLoRA)**
```bash
cd fine_tuning
python train_transformers_peft.py --config ../configs/config.yaml
```

**Option B: Using Unsloth (2-5x faster)**
```bash
cd fine_tuning
python train_unsloth_simple.py --config ../configs/config.yaml
```

### 3. Run Inference

```python
from inference.inference_engines import get_engine

# Initialize engine
engine = get_engine(
    engine_type="transformers",
    model_path="results/checkpoints/final"
)

# Generate response
response = engine.generate("Explain quantum computing in simple terms.")
print(response)
```

### 4. Evaluate & Benchmark

```bash
cd evaluation
python complete_evaluation.py --config ../configs/config.yaml
```

---

## Framework Comparison

### Fine-tuning Methods

| Feature | Transformers + PEFT | Unsloth |
|---------|-------------------|---------|
| Speed | 1x (baseline) | 2-5x faster |
| Memory | ~10GB | ~6-8GB |
| QLoRA Support | Yes | Yes |
| Flash Attention | Manual | Automatic |
| Export Formats | HuggingFace | HuggingFace, GGUF |

### Inference Frameworks

| Feature | Transformers | Unsloth | vLLM | Ollama |
|---------|-------------|---------|------|--------|
| Latency | Medium | Low | Very Low | Low |
| Throughput | Low | Medium | Very High | Medium |
| Memory | Medium | Low | Medium | Low |
| Batching | Manual | Manual | Continuous | Single |
| Ease of Use | 4/5 | 3/5 | 3/5 | 5/5 |
| Production Ready | 3/5 | 2/5 | 5/5 | 4/5 |

---

## Usage Guide

### Configuration

All settings are managed through `configs/config.yaml`:

```yaml
model:
  base_model: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
  
lora:
  rank: 16
  alpha: 32
  dropout: 0.05

training:
  epochs: 3
  batch_size: 4
  learning_rate: 2.0e-4
  
quantization:
  bits: 4
  type: "nf4"  # Normal Float 4
```

### Inference API

```python
from inference.inference_engines import get_engine

# Available engines: "transformers", "unsloth", "vllm", "ollama"
engine = get_engine(
    engine_type="vllm",
    model_path="results/checkpoints/final",
    max_tokens=512,
    temperature=0.7
)

# Single generation
response = engine.generate("What is machine learning?")

# Batch generation
responses = engine.generate_batch([
    "Explain AI",
    "What is deep learning?",
    "How does NLP work?"
])

# Cleanup
engine.cleanup()
```

### Evaluation

```python
from evaluation.evaluate_model import evaluate_model

# Evaluate model predictions
results = evaluate_model(
    model_path="results/checkpoints/final",
    test_data_path="data/processed/test.json",
    num_samples=50
)

print(f"BLEU: {results['bleu']:.4f}")
print(f"ROUGE-L: {results['rouge_l']:.4f}")
print(f"BERTScore: {results['bertscore_f1']:.4f}")
```

---

## Evaluation

### Automatic Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| BLEU | N-gram precision overlap | 0-100 |
| ROUGE-L | Longest common subsequence | 0-1 |
| BERTScore | Semantic similarity | 0-1 |

### Human Evaluation

Responses are manually evaluated on:
- **Helpfulness** (1-5): How useful is the response?
- **Factuality** (1-5): Is the information accurate?
- **Instruction Following** (1-5): Does it follow instructions?

### Benchmark Metrics

- **Latency**: Time to first token (ms)
- **Throughput**: Tokens per second
- **Memory Usage**: Peak GPU memory (GB)

---

## Results

Results are saved in `results/` after running the evaluation pipeline.

### Expected Results Structure

```
results/
├── benchmarks/
│   ├── transformers_benchmark.json
│   ├── unsloth_benchmark.json
│   ├── vllm_benchmark.json
│   └── ollama_benchmark.json
├── evaluation/
│   ├── transformers_eval.json
│   ├── unsloth_eval.json
│   ├── before_after_comparison.json
│   └── human_eval_completed.json
└── checkpoints/
    └── final/                      # Final merged model
```

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---
