from typing import Dict, List, Any
import logging
from .metrics import QualityMetrics
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)

class QualityFilter:
    """Filters and validates generated instruction data based on quality metrics."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the quality filter.
        
        Args:
            config: Configuration dictionary containing quality thresholds
        """
        self.config = config
        self.metrics = QualityMetrics()
        self.min_quality_score = config["agent"]["quality_control"]["min_quality_score"]
        
    def filter_qa_pairs(self, qa_pairs: List[Dict[str, Any]], contexts: Dict[str, str]) -> List[Dict[str, Any]]:
        """
        Filter QA pairs based on quality metrics.
        
        Args:
            qa_pairs: List of question-answer pairs to filter
            contexts: Dictionary mapping source to original context
            
        Returns:
            List of filtered QA pairs that meet quality standards
        """
        filtered_pairs = []
        metrics_by_source = defaultdict(list)
        
        # Compute metrics for each QA pair
        for qa_pair in qa_pairs:
            try:
                source = qa_pair["source"]
                context = contexts.get(source, "")
                
                if not context:
                    logger.warning(f"No context found for source: {source}")
                    continue
                
                # Compute quality metrics
                metrics = self.metrics.compute_metrics(qa_pair, context)
                qa_pair["metrics"] = metrics
                metrics_by_source[source].append(qa_pair)
                
            except Exception as e:
                logger.error(f"Error computing metrics for QA pair: {e}")
                continue
        
        # Filter and select best pairs for each source
        for source, source_pairs in metrics_by_source.items():
            try:
                # Evaluate diversity within source
                diversity_metrics = self.metrics.evaluate_diversity(source_pairs)
                
                # Sort pairs by quality score
                sorted_pairs = sorted(
                    source_pairs,
                    key=lambda x: x["metrics"]["overall_quality"],
                    reverse=True
                )
                
                # Filter based on minimum quality score
                quality_pairs = [
                    pair for pair in sorted_pairs
                    if pair["metrics"]["overall_quality"] >= self.min_quality_score
                ]
                
                # Select diverse subset
                selected_pairs = self._select_diverse_subset(quality_pairs)
                filtered_pairs.extend(selected_pairs)
                
            except Exception as e:
                logger.error(f"Error filtering pairs for source {source}: {e}")
                continue
        
        return filtered_pairs
    
    def _select_diverse_subset(self, qa_pairs: List[Dict[str, Any]], max_pairs: int = None) -> List[Dict[str, Any]]:
        """
        Select a diverse subset of QA pairs using greedy selection.
        
        Args:
            qa_pairs: List of QA pairs to select from
            max_pairs: Maximum number of pairs to select (optional)
            
        Returns:
            List of selected diverse QA pairs
        """
        if not qa_pairs:
            return []
            
        if max_pairs is None:
            max_pairs = len(qa_pairs)
        
        selected = [qa_pairs[0]]  # Start with highest quality pair
        remaining = qa_pairs[1:]
        
        while len(selected) < max_pairs and remaining:
            # Find pair with maximum diversity from selected pairs
            max_diversity = -1
            best_pair_idx = -1
            
            for i, pair in enumerate(remaining):
                # Calculate average diversity with selected pairs
                diversity = self._compute_pair_diversity(pair, selected)
                
                if diversity > max_diversity:
                    max_diversity = diversity
                    best_pair_idx = i
            
            if best_pair_idx >= 0:
                selected.append(remaining.pop(best_pair_idx))
            else:
                break
        
        return selected
    
    def _compute_pair_diversity(self, pair: Dict[str, Any], selected_pairs: List[Dict[str, Any]]) -> float:
        """
        Compute diversity score between a pair and already selected pairs.
        
        Args:
            pair: QA pair to evaluate
            selected_pairs: List of already selected pairs
            
        Returns:
            Average diversity score
        """
        diversities = []
        
        for selected in selected_pairs:
            # Compare questions
            question_diversity = self.metrics._compute_diversity_score(
                [pair["question"], selected["question"]]
            )
            
            # Compare answers
            answer_diversity = self.metrics._compute_diversity_score(
                [pair["answer"], selected["answer"]]
            )
            
            # Combine scores
            pair_diversity = (question_diversity + answer_diversity) / 2
            diversities.append(pair_diversity)
        
        return np.mean(diversities) if diversities else 0.0
    
    def validate_against_source(self, qa_pairs: List[Dict[str, Any]], contexts: Dict[str, str]) -> List[Dict[str, Any]]:
        """
        Validate that answers can be derived from source context.
        
        Args:
            qa_pairs: List of QA pairs to validate
            contexts: Dictionary mapping source to original context
            
        Returns:
            List of validated QA pairs
        """
        validated_pairs = []
        
        for qa_pair in qa_pairs:
            try:
                source = qa_pair["source"]
                context = contexts.get(source, "")
                
                if not context:
                    continue
                
                # Check if answer is contained in context (basic validation)
                answer = qa_pair["answer"].lower()
                context_lower = context.lower()
                
                # Compute relevance and similarity metrics
                metrics = qa_pair.get("metrics", {})
                relevance = metrics.get("relevance", 0.0)
                rouge_l = metrics.get("rougeL", 0.0)
                
                # Validate based on multiple criteria
                is_valid = (
                    relevance >= 0.5 and  # High relevance to context
                    rouge_l >= 0.3 and    # Significant overlap with context
                    len(answer.split()) >= 3  # Minimum answer length
                )
                
                if is_valid:
                    validated_pairs.append(qa_pair)
                    
            except Exception as e:
                logger.error(f"Error validating QA pair: {e}")
                continue
        
        return validated_pairs