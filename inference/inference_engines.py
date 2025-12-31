"""
NLP2 Final Project - Unified Inference Module
=============================================
Unified interface for running inference across all 4 frameworks:
- Transformers
- Unsloth
- vLLM
- Ollama

This module provides a consistent API for benchmarking and evaluation.
"""

import os
import sys
import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field

import torch
import yaml


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


@dataclass
class InferenceResult:
    """Container for inference results."""
    text: str
    prompt: str
    tokens_generated: int
    latency_seconds: float
    tokens_per_second: float
    memory_used_gb: float = 0.0
    metadata: Dict = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    framework: str
    model_path: str
    num_requests: int
    avg_latency_seconds: float
    avg_throughput_tokens_per_sec: float
    peak_memory_gb: float
    total_tokens_generated: int
    total_time_seconds: float
    individual_results: List[InferenceResult] = field(default_factory=list)


class BaseInferenceEngine(ABC):
    """Abstract base class for inference engines."""
    
    def __init__(self, config: Optional[dict] = None, config_path: str = "configs/config.yaml"):
        self.config = config or load_config(config_path)
        self.model = None
        self.tokenizer = None
        self.framework_name = "base"
        
    @abstractmethod
    def load_model(self, model_path: str):
        """Load the model for inference."""
        pass
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> InferenceResult:
        """Generate text from a prompt."""
        pass
    
    def generate_batch(
        self,
        prompts: List[str],
        **kwargs
    ) -> List[InferenceResult]:
        """Generate text for multiple prompts."""
        return [self.generate(p, **kwargs) for p in prompts]
    
    def benchmark(
        self,
        prompts: List[str],
        num_warmup: int = 3,
        **kwargs
    ) -> BenchmarkResult:
        """Run benchmark on a list of prompts."""
        # Warmup runs
        print(f" Warming up ({num_warmup} runs)...")
        for i in range(min(num_warmup, len(prompts))):
            _ = self.generate(prompts[i % len(prompts)], **kwargs)
        
        # Reset memory stats
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        # Benchmark runs
        print(f"Benchmarking ({len(prompts)} prompts)...")
        results = []
        total_start = time.time()
        
        for i, prompt in enumerate(prompts):
            result = self.generate(prompt, **kwargs)
            results.append(result)
            if (i + 1) % 10 == 0:
                print(f"Completed {i + 1}/{len(prompts)}")
        
        total_time = time.time() - total_start
        
        # Calculate statistics
        peak_memory = 0
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / 1024**3
        
        total_tokens = sum(r.tokens_generated for r in results)
        avg_latency = sum(r.latency_seconds for r in results) / len(results)
        avg_throughput = sum(r.tokens_per_second for r in results) / len(results)
        
        return BenchmarkResult(
            framework=self.framework_name,
            model_path=str(getattr(self, 'model_path', 'unknown')),
            num_requests=len(prompts),
            avg_latency_seconds=avg_latency,
            avg_throughput_tokens_per_sec=avg_throughput,
            peak_memory_gb=peak_memory,
            total_tokens_generated=total_tokens,
            total_time_seconds=total_time,
            individual_results=results
        )
    
    def cleanup(self):
        """Clean up resources."""
        if self.model is not None:
            del self.model
        if self.tokenizer is not None:
            del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ============================================================================
# TRANSFORMERS INFERENCE
# ============================================================================

class TransformersInference(BaseInferenceEngine):
    """Inference using HuggingFace Transformers."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.framework_name = "transformers"
        
    def load_model(
        self,
        model_path: str,
        use_4bit: bool = True,
        device_map: str = "auto"
    ):
        """Load model with optional quantization."""
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        
        print(f" Loading Transformers model: {model_path}")
        self.model_path = model_path
        
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Quantization config
        if use_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quant_config,
                device_map=device_map,
                torch_dtype=torch.bfloat16
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device_map,
                torch_dtype=torch.bfloat16
            )
        
        self.model.eval()
        print(f"Model loaded on {self.model.device}")
        
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
        **kwargs
    ) -> InferenceResult:
        """Generate text using Transformers."""
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_length = inputs.input_ids.shape[1]
        
        # Generate
        start_time = time.time()
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        latency = time.time() - start_time
        
        # Decode
        generated_tokens = outputs[0][input_length:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        tokens_generated = len(generated_tokens)
        
        # Memory
        memory_used = 0
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1024**3
        
        return InferenceResult(
            text=generated_text,
            prompt=prompt,
            tokens_generated=tokens_generated,
            latency_seconds=latency,
            tokens_per_second=tokens_generated / latency if latency > 0 else 0,
            memory_used_gb=memory_used
        )


# ============================================================================
# UNSLOTH INFERENCE
# ============================================================================

class UnslothInference(BaseInferenceEngine):
    """Inference using Unsloth's optimized implementation."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.framework_name = "unsloth"
        
    def load_model(self, model_path: str, max_seq_length: int = 2048):
        """Load model with Unsloth optimizations."""
        from unsloth import FastLanguageModel
        
        print(f" Loading Unsloth model: {model_path}")
        self.model_path = model_path
        
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=max_seq_length,
            dtype=None,  # Auto-detect
            load_in_4bit=True,
        )
        
        # Enable fast inference
        FastLanguageModel.for_inference(self.model)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        print(f"✓ Model loaded with Unsloth optimizations")
        
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
        **kwargs
    ) -> InferenceResult:
        """Generate text using Unsloth."""
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_length = inputs.input_ids.shape[1]
        
        # Generate
        start_time = time.time()
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            use_cache=True,
        )
        
        latency = time.time() - start_time
        
        # Decode
        generated_tokens = outputs[0][input_length:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        tokens_generated = len(generated_tokens)
        
        # Memory
        memory_used = 0
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1024**3
        
        return InferenceResult(
            text=generated_text,
            prompt=prompt,
            tokens_generated=tokens_generated,
            latency_seconds=latency,
            tokens_per_second=tokens_generated / latency if latency > 0 else 0,
            memory_used_gb=memory_used
        )


# ============================================================================
# vLLM INFERENCE
# ============================================================================

class VLLMInference(BaseInferenceEngine):
    """Inference using vLLM for high-throughput serving."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.framework_name = "vllm"
        self.llm = None
        
    def load_model(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 2048
    ):
        """Load model with vLLM."""
        from vllm import LLM, SamplingParams
        
        print(f" Loading vLLM model: {model_path}")
        self.model_path = model_path
        
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=True,
            dtype="bfloat16"
        )
        
        print(f"✓ Model loaded with vLLM optimizations")
        
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        **kwargs
    ) -> InferenceResult:
        """Generate text using vLLM."""
        from vllm import SamplingParams
        
        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )
        
        start_time = time.time()
        outputs = self.llm.generate([prompt], sampling_params)
        latency = time.time() - start_time
        
        output = outputs[0]
        generated_text = output.outputs[0].text
        tokens_generated = len(output.outputs[0].token_ids)
        
        # Memory
        memory_used = 0
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1024**3
        
        return InferenceResult(
            text=generated_text,
            prompt=prompt,
            tokens_generated=tokens_generated,
            latency_seconds=latency,
            tokens_per_second=tokens_generated / latency if latency > 0 else 0,
            memory_used_gb=memory_used
        )
    
    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> List[InferenceResult]:
        """Generate text for multiple prompts (batched for efficiency)."""
        from vllm import SamplingParams
        
        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        
        start_time = time.time()
        outputs = self.llm.generate(prompts, sampling_params)
        total_latency = time.time() - start_time
        
        results = []
        for i, output in enumerate(outputs):
            generated_text = output.outputs[0].text
            tokens_generated = len(output.outputs[0].token_ids)
            per_request_latency = total_latency / len(prompts)
            
            results.append(InferenceResult(
                text=generated_text,
                prompt=prompts[i],
                tokens_generated=tokens_generated,
                latency_seconds=per_request_latency,
                tokens_per_second=tokens_generated / per_request_latency if per_request_latency > 0 else 0,
            ))
        
        return results


# ============================================================================
# OLLAMA INFERENCE
# ============================================================================

class OllamaInference(BaseInferenceEngine):
    """Inference using Ollama for local model serving."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.framework_name = "ollama"
        self.model_name = None
        
    def load_model(self, model_name: str, create_from_gguf: Optional[str] = None):
        """
        Load model with Ollama.
        
        Args:
            model_name: Name of the model in Ollama (e.g., "tinyllama-finetuned")
            create_from_gguf: Path to GGUF file to create model from
        """
        import subprocess
        
        print(f" Loading Ollama model: {model_name}")
        self.model_name = model_name
        self.model_path = model_name
        
        # Create model from GGUF if provided
        if create_from_gguf:
            self._create_model_from_gguf(model_name, create_from_gguf)
        
        # Verify model exists
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True
            )
            if model_name not in result.stdout:
                print(f"⚠ Model {model_name} not found in Ollama. Pull it first:")
                print(f"ollama pull {model_name}")
            else:
                print(f"✓ Model {model_name} is available")
        except FileNotFoundError:
            print(f"Ollama not installed. Install from: https://ollama.ai")
            
    def _create_model_from_gguf(self, model_name: str, gguf_path: str):
        """Create Ollama model from GGUF file."""
        import subprocess
        import tempfile
        
        modelfile_content = f"""FROM {gguf_path}
TEMPLATE \"\"\"<|system|>
You are a helpful assistant.</s>
<|user|>
{{{{ .Prompt }}}}</s>
<|assistant|>
\"\"\"
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER stop </s>
PARAMETER stop <|user|>
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.Modelfile', delete=False) as f:
            f.write(modelfile_content)
            modelfile_path = f.name
        
        try:
            subprocess.run(
                ["ollama", "create", model_name, "-f", modelfile_path],
                check=True
            )
            print(f"✓ Created Ollama model: {model_name}")
        finally:
            os.unlink(modelfile_path)
        
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> InferenceResult:
        """Generate text using Ollama."""
        import requests
        
        start_time = time.time()
        
        # Use Ollama API
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": self.model_name,
                "prompt": prompt,
                "options": {
                    "num_predict": max_new_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                },
                "stream": False
            }
        )
        
        latency = time.time() - start_time
        
        if response.status_code != 200:
            raise RuntimeError(f"Ollama API error: {response.text}")
        
        result = response.json()
        generated_text = result.get("response", "")
        
        # Estimate tokens (Ollama doesn't always return token count)
        tokens_generated = result.get("eval_count", len(generated_text.split()))
        
        return InferenceResult(
            text=generated_text,
            prompt=prompt,
            tokens_generated=tokens_generated,
            latency_seconds=latency,
            tokens_per_second=tokens_generated / latency if latency > 0 else 0,
            metadata={
                "total_duration": result.get("total_duration", 0),
                "load_duration": result.get("load_duration", 0),
                "eval_duration": result.get("eval_duration", 0),
            }
        )


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def get_inference_engine(framework: str, **kwargs) -> BaseInferenceEngine:
    """Factory function to get inference engine by name."""
    engines = {
        "transformers": TransformersInference,
        "unsloth": UnslothInference,
        "vllm": VLLMInference,
        "ollama": OllamaInference,
    }
    
    if framework.lower() not in engines:
        raise ValueError(f"Unknown framework: {framework}. Choose from: {list(engines.keys())}")
    
    return engines[framework.lower()](**kwargs)


# ============================================================================
# MAIN - Interactive Testing
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test inference engines")
    parser.add_argument("--framework", type=str, default="transformers",
                       choices=["transformers", "unsloth", "vllm", "ollama"])
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="What is machine learning?")
    
    args = parser.parse_args()
    
    engine = get_inference_engine(args.framework)
    engine.load_model(args.model_path)
    
    result = engine.generate(args.prompt)
    
    print(f"\n{'='*60}")
    print(f"Framework: {args.framework}")
    print(f"Model: {args.model_path}")
    print(f"{'='*60}")
    print(f"Prompt: {args.prompt}")
    print(f"Response: {result.text}")
    print(f"{'='*60}")
    print(f"Tokens: {result.tokens_generated}")
    print(f"Latency: {result.latency_seconds:.3f}s")
    print(f"Throughput: {result.tokens_per_second:.1f} tokens/sec")
    print(f"Memory: {result.memory_used_gb:.2f} GB")
