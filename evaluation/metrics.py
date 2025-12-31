"""
NLP2 Final Project - Evaluation Module
======================================
Implements automatic evaluation metrics:
- BLEU: n-gram overlap
- ROUGE-L: longest common subsequence
- BERTScore: semantic similarity

Also provides tools for human evaluation.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict

import torch
import yaml
import numpy as np
from tqdm import tqdm


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    bleu: float = 0.0
    bleu_1: float = 0.0
    bleu_2: float = 0.0
    bleu_3: float = 0.0
    bleu_4: float = 0.0
    rouge_l_precision: float = 0.0
    rouge_l_recall: float = 0.0
    rouge_l_f1: float = 0.0
    bertscore_precision: float = 0.0
    bertscore_recall: float = 0.0
    bertscore_f1: float = 0.0
    num_samples: int = 0
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class HumanEvalSample:
    """Container for human evaluation sample."""
    prompt: str
    reference: str
    generated: str
    helpfulness: int = 0  # 1-5
    factuality: int = 0   # 1-5
    instruction_following: int = 0  # 1-5
    notes: str = ""


class MetricsCalculator:
    """
    Calculates evaluation metrics for generated text.
    
    Metrics:
    - BLEU (1-4 gram): measures n-gram overlap
    - ROUGE-L: measures longest common subsequence
    - BERTScore: measures semantic similarity using BERT embeddings
    """
    
    def __init__(self, config: Optional[dict] = None, config_path: str = "configs/config.yaml"):
        self.config = config or load_config(config_path)
        self._bleu_scorer = None
        self._rouge_scorer = None
        self._bertscore_model = None
        
    def _init_bleu(self):
        """Initialize BLEU scorer."""
        if self._bleu_scorer is None:
            from nltk.translate.bleu_score import SmoothingFunction
            self._smoothing = SmoothingFunction()
        
    def _init_rouge(self):
        """Initialize ROUGE scorer."""
        if self._rouge_scorer is None:
            from rouge_score import rouge_scorer
            self._rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    def _init_bertscore(self):
        """Initialize BERTScore."""
        if self._bertscore_model is None:
            # BERTScore will be initialized on first use
            pass
    
    def calculate_bleu(
        self,
        predictions: List[str],
        references: List[str],
        max_n: int = 4
    ) -> Dict[str, float]:
        """
        Calculate BLEU scores.
        
        Args:
            predictions: List of generated texts
            references: List of reference texts
            max_n: Maximum n-gram size (default: 4)
            
        Returns:
            Dictionary with BLEU scores
        """
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        from nltk.tokenize import word_tokenize
        import nltk
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab', quiet=True)
        
        smoothing = SmoothingFunction()
        
        bleu_scores = {f"bleu_{i}": [] for i in range(1, max_n + 1)}
        bleu_scores["bleu"] = []
        
        for pred, ref in zip(predictions, references):
            # Tokenize
            pred_tokens = word_tokenize(pred.lower())
            ref_tokens = [word_tokenize(ref.lower())]
            
            # Calculate individual n-gram BLEU
            for n in range(1, max_n + 1):
                weights = tuple([1.0/n] * n + [0.0] * (4 - n))
                try:
                    score = sentence_bleu(
                        ref_tokens,
                        pred_tokens,
                        weights=weights,
                        smoothing_function=smoothing.method1
                    )
                except:
                    score = 0.0
                bleu_scores[f"bleu_{n}"].append(score)
            
            # Calculate cumulative BLEU-4
            try:
                bleu_4 = sentence_bleu(
                    ref_tokens,
                    pred_tokens,
                    smoothing_function=smoothing.method1
                )
            except:
                bleu_4 = 0.0
            bleu_scores["bleu"].append(bleu_4)
        
        # Average scores
        return {k: np.mean(v) for k, v in bleu_scores.items()}
    
    def calculate_rouge(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """
        Calculate ROUGE-L scores.
        
        Args:
            predictions: List of generated texts
            references: List of reference texts
            
        Returns:
            Dictionary with ROUGE-L precision, recall, and F1
        """
        from rouge_score import rouge_scorer
        
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        
        precision_scores = []
        recall_scores = []
        f1_scores = []
        
        for pred, ref in zip(predictions, references):
            scores = scorer.score(ref, pred)
            precision_scores.append(scores['rougeL'].precision)
            recall_scores.append(scores['rougeL'].recall)
            f1_scores.append(scores['rougeL'].fmeasure)
        
        return {
            "rouge_l_precision": np.mean(precision_scores),
            "rouge_l_recall": np.mean(recall_scores),
            "rouge_l_f1": np.mean(f1_scores)
        }
    
    def calculate_bertscore(
        self,
        predictions: List[str],
        references: List[str],
        model_type: str = None,
        batch_size: int = 32
    ) -> Dict[str, float]:
        """
        Calculate BERTScore for semantic similarity.
        
        Args:
            predictions: List of generated texts
            references: List of reference texts
            model_type: BERT model to use (default from config)
            batch_size: Batch size for processing
            
        Returns:
            Dictionary with BERTScore precision, recall, and F1
        """
        from bert_score import score
        
        model_type = model_type or self.config["evaluation"].get(
            "bertscore_model", "microsoft/deberta-xlarge-mnli"
        )
        
        print(f"Calculating BERTScore using {model_type}...")
        
        P, R, F1 = score(
            predictions,
            references,
            model_type=model_type,
            batch_size=batch_size,
            verbose=False,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        return {
            "bertscore_precision": P.mean().item(),
            "bertscore_recall": R.mean().item(),
            "bertscore_f1": F1.mean().item()
        }
    
    def calculate_all_metrics(
        self,
        predictions: List[str],
        references: List[str],
        include_bertscore: bool = True
    ) -> EvaluationResult:
        """
        Calculate all evaluation metrics.
        
        Args:
            predictions: List of generated texts
            references: List of reference texts
            include_bertscore: Whether to include BERTScore (slower)
            
        Returns:
            EvaluationResult with all metrics
        """
        print(f" Calculating metrics for {len(predictions)} samples...")
        
        # BLEU
        print(f"Computing BLEU scores...")
        bleu_scores = self.calculate_bleu(predictions, references)
        
        # ROUGE
        print(f"Computing ROUGE-L scores...")
        rouge_scores = self.calculate_rouge(predictions, references)
        
        # BERTScore (optional, slower)
        bertscore_results = {
            "bertscore_precision": 0.0,
            "bertscore_recall": 0.0,
            "bertscore_f1": 0.0
        }
        if include_bertscore:
            print(f"Computing BERTScore...")
            bertscore_results = self.calculate_bertscore(predictions, references)
        
        return EvaluationResult(
            bleu=bleu_scores["bleu"],
            bleu_1=bleu_scores["bleu_1"],
            bleu_2=bleu_scores["bleu_2"],
            bleu_3=bleu_scores["bleu_3"],
            bleu_4=bleu_scores["bleu_4"],
            rouge_l_precision=rouge_scores["rouge_l_precision"],
            rouge_l_recall=rouge_scores["rouge_l_recall"],
            rouge_l_f1=rouge_scores["rouge_l_f1"],
            bertscore_precision=bertscore_results["bertscore_precision"],
            bertscore_recall=bertscore_results["bertscore_recall"],
            bertscore_f1=bertscore_results["bertscore_f1"],
            num_samples=len(predictions)
        )


class ModelEvaluator:
    """
    Evaluates model performance before and after fine-tuning.
    """
    
    def __init__(
        self,
        config: Optional[dict] = None,
        config_path: str = "configs/config.yaml"
    ):
        self.config = config or load_config(config_path)
        self.metrics_calculator = MetricsCalculator(config=self.config)
        
    def load_test_data(self, test_data_path: str = "data/processed/eval_dataset/test_eval.json"):
        """Load test dataset with prompts and references."""
        print(f" Loading test data from {test_data_path}")
        
        import json
        with open(test_data_path, "r") as f:
            data = [json.loads(line) for line in f]
        
        print(f"Loaded {len(data)} test samples")
        return data
    
    def generate_responses(
        self,
        inference_engine,
        test_data: List[Dict],
        max_samples: Optional[int] = None,
        **generate_kwargs
    ) -> List[str]:
        """Generate responses for test prompts."""
        from inference.inference_engines import BaseInferenceEngine
        
        if max_samples:
            test_data = test_data[:max_samples]
        
        print(f"🤖 Generating {len(test_data)} responses...")
        
        predictions = []
        for i, sample in enumerate(tqdm(test_data)):
            prompt = sample["prompt"]
            result = inference_engine.generate(prompt, **generate_kwargs)
            predictions.append(result.text)
            
            if (i + 1) % 50 == 0:
                print(f"Generated {i + 1}/{len(test_data)}")
        
        return predictions
    
    def evaluate_model(
        self,
        inference_engine,
        test_data_path: str = "data/processed/eval_dataset/test_eval.json",
        max_samples: Optional[int] = None,
        include_bertscore: bool = True,
        save_results: bool = True,
        output_path: Optional[str] = None,
        model_name: str = "model",
        **generate_kwargs
    ) -> Tuple[EvaluationResult, List[Dict]]:
        """
        Run full evaluation pipeline.
        
        Args:
            inference_engine: Loaded inference engine
            test_data_path: Path to test data
            max_samples: Maximum samples to evaluate (None for all)
            include_bertscore: Whether to calculate BERTScore
            save_results: Whether to save results to file
            output_path: Path to save results
            model_name: Name for the model in results
            
        Returns:
            Tuple of (EvaluationResult, list of sample results)
        """
        # Load test data
        test_data = self.load_test_data(test_data_path)
        
        if max_samples:
            test_data = test_data[:max_samples]
        
        # Generate predictions
        predictions = self.generate_responses(
            inference_engine, test_data, **generate_kwargs
        )
        
        # Extract references
        references = [sample["reference"] for sample in test_data]
        
        # Calculate metrics
        results = self.metrics_calculator.calculate_all_metrics(
            predictions, references, include_bertscore=include_bertscore
        )
        
        # Create detailed results
        sample_results = []
        for i, (sample, pred) in enumerate(zip(test_data, predictions)):
            sample_results.append({
                "id": i,
                "instruction": sample.get("instruction", ""),
                "context": sample.get("context", ""),
                "reference": sample["reference"],
                "prediction": pred,
                "category": sample.get("category", "")
            })
        
        # Save results
        if save_results:
            output_path = output_path or f"results/{model_name}_evaluation.json"
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, "w") as f:
                json.dump({
                    "model": model_name,
                    "metrics": results.to_dict(),
                    "samples": sample_results
                }, f, indent=2)
            
            print(f" Results saved to {output_path}")
        
        return results, sample_results
    
    def compare_models(
        self,
        results: Dict[str, EvaluationResult],
        output_path: str = "results/comparison.json"
    ) -> Dict:
        """
        Compare evaluation results across multiple models.
        
        Args:
            results: Dictionary mapping model names to EvaluationResult
            output_path: Path to save comparison
            
        Returns:
            Comparison dictionary
        """
        comparison = {
            "models": list(results.keys()),
            "metrics": {}
        }
        
        metrics = ["bleu", "rouge_l_f1", "bertscore_f1"]
        
        for metric in metrics:
            comparison["metrics"][metric] = {
                name: getattr(result, metric)
                for name, result in results.items()
            }
        
        # Find best model for each metric
        comparison["best"] = {}
        for metric in metrics:
            best_model = max(
                comparison["metrics"][metric].items(),
                key=lambda x: x[1]
            )[0]
            comparison["best"][metric] = best_model
        
        # Save comparison
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(comparison, f, indent=2)
        
        return comparison


class HumanEvaluator:
    """
    Tools for human evaluation of model outputs.
    """
    
    def __init__(self, num_samples: int = 5):
        self.num_samples = num_samples
        self.samples = []
        
    def prepare_samples(
        self,
        test_data: List[Dict],
        predictions: List[str],
        random_seed: int = 42
    ) -> List[HumanEvalSample]:
        """Prepare samples for human evaluation."""
        import random
        random.seed(random_seed)
        
        # Select random samples
        indices = random.sample(range(len(test_data)), min(self.num_samples, len(test_data)))
        
        self.samples = []
        for idx in indices:
            sample = HumanEvalSample(
                prompt=test_data[idx].get("instruction", ""),
                reference=test_data[idx]["reference"],
                generated=predictions[idx]
            )
            self.samples.append(sample)
        
        return self.samples
    
    def save_for_annotation(self, output_path: str = "results/human_eval_samples.json"):
        """Save samples for human annotation."""
        samples_dict = []
        for i, sample in enumerate(self.samples):
            samples_dict.append({
                "id": i,
                "prompt": sample.prompt,
                "reference": sample.reference,
                "generated": sample.generated,
                "helpfulness": 0,
                "factuality": 0,
                "instruction_following": 0,
                "notes": ""
            })
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(samples_dict, f, indent=2)
        
        print(f" Human evaluation samples saved to {output_path}")
        print(f"Please rate each sample on a scale of 1-5 for:")
        print(f"- Helpfulness: How helpful is the response?")
        print(f"- Factuality: Is the information accurate?")
        print(f"- Instruction-following: Does it follow the instruction?")
        
        return output_path
    
    def load_annotations(self, annotations_path: str) -> List[HumanEvalSample]:
        """Load annotated samples."""
        with open(annotations_path, "r") as f:
            data = json.load(f)
        
        self.samples = []
        for item in data:
            sample = HumanEvalSample(
                prompt=item["prompt"],
                reference=item["reference"],
                generated=item["generated"],
                helpfulness=item["helpfulness"],
                factuality=item["factuality"],
                instruction_following=item["instruction_following"],
                notes=item.get("notes", "")
            )
            self.samples.append(sample)
        
        return self.samples
    
    def calculate_human_scores(self) -> Dict[str, float]:
        """Calculate average human evaluation scores."""
        if not self.samples:
            return {}
        
        scores = {
            "helpfulness": np.mean([s.helpfulness for s in self.samples if s.helpfulness > 0]),
            "factuality": np.mean([s.factuality for s in self.samples if s.factuality > 0]),
            "instruction_following": np.mean([s.instruction_following for s in self.samples if s.instruction_following > 0]),
        }
        
        scores["average"] = np.mean(list(scores.values()))
        
        return scores


def print_evaluation_results(result: EvaluationResult, model_name: str = "Model"):
    """Pretty print evaluation results."""
    print(f"\n{'='*60}")
    print(f" EVALUATION RESULTS: {model_name}")
    print(f"{'='*60}")
    print(f"\n   Samples evaluated: {result.num_samples}")
    
    print(f"\n   BLEU Scores:")
    print(f"BLEU-1:    {result.bleu_1:.4f}")
    print(f"BLEU-2:    {result.bleu_2:.4f}")
    print(f"BLEU-3:    {result.bleu_3:.4f}")
    print(f"BLEU-4:    {result.bleu_4:.4f}")
    print(f"BLEU (avg): {result.bleu:.4f}")
    
    print(f"\n   ROUGE-L Scores:")
    print(f"Precision: {result.rouge_l_precision:.4f}")
    print(f"Recall:    {result.rouge_l_recall:.4f}")
    print(f"F1:        {result.rouge_l_f1:.4f}")
    
    if result.bertscore_f1 > 0:
        print(f"\n   BERTScore:")
        print(f"Precision: {result.bertscore_precision:.4f}")
        print(f"Recall:    {result.bertscore_recall:.4f}")
        print(f"F1:        {result.bertscore_f1:.4f}")
    
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    # Test metrics calculator
    calc = MetricsCalculator()
    
    predictions = [
        "Machine learning is a subset of artificial intelligence.",
        "Python is a programming language used for data science."
    ]
    references = [
        "Machine learning is a branch of AI that enables computers to learn.",
        "Python is a popular programming language for data analysis."
    ]
    
    results = calc.calculate_all_metrics(predictions, references, include_bertscore=False)
    print_evaluation_results(results, "Test Model")
