from typing import Dict, List, Any, Tuple
import logging
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from nltk.tokenize import word_tokenize
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)

class QualityMetrics:
    """Evaluates the quality of generated instruction data using various metrics."""
    
    def __init__(self):
        """Initialize quality metrics."""
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smooth = SmoothingFunction().method1
        
    def compute_metrics(self, qa_pair: Dict[str, Any], context: str) -> Dict[str, float]:
        """
        Compute quality metrics for a question-answer pair.
        
        Args:
            qa_pair: Dictionary containing question and answer
            context: Original context used to generate the QA pair
            
        Returns:
            Dictionary of metric scores
        """
        question = qa_pair["question"]
        answer = qa_pair["answer"]
        
        try:
            # Compute BLEU score between answer and context
            bleu_score = self._compute_bleu(answer, context)
            
            # Compute METEOR score
            meteor = self._compute_meteor(answer, context)
            
            # Compute ROUGE scores
            rouge_scores = self._compute_rouge(answer, context)
            
            # Compute answer relevance score
            relevance = self._compute_relevance(answer, context)
            
            # Compute question quality score
            question_quality = self._evaluate_question_quality(question)
            
            return {
                "bleu": bleu_score,
                "meteor": meteor,
                "rouge1": rouge_scores["rouge1"],
                "rouge2": rouge_scores["rouge2"],
                "rougeL": rouge_scores["rougeL"],
                "relevance": relevance,
                "question_quality": question_quality,
                "overall_quality": self._compute_overall_score([
                    bleu_score,
                    meteor,
                    rouge_scores["rougeL"],
                    relevance,
                    question_quality
                ])
            }
            
        except Exception as e:
            logger.error(f"Error computing metrics: {e}")
            return defaultdict(float)
    
    def _compute_bleu(self, answer: str, context: str) -> float:
        """Compute BLEU score between answer and context."""
        try:
            reference = [word_tokenize(context.lower())]
            candidate = word_tokenize(answer.lower())
            return sentence_bleu(reference, candidate, smoothing_function=self.smooth)
        except Exception as e:
            logger.error(f"Error computing BLEU score: {e}")
            return 0.0
    
    def _compute_meteor(self, answer: str, context: str) -> float:
        """Compute METEOR score between answer and context."""
        try:
            reference = word_tokenize(context.lower())
            candidate = word_tokenize(answer.lower())
            return meteor_score([reference], candidate)
        except Exception as e:
            logger.error(f"Error computing METEOR score: {e}")
            return 0.0
    
    def _compute_rouge(self, answer: str, context: str) -> Dict[str, float]:
        """Compute ROUGE scores between answer and context."""
        try:
            scores = self.scorer.score(context, answer)
            return {
                "rouge1": scores["rouge1"].fmeasure,
                "rouge2": scores["rouge2"].fmeasure,
                "rougeL": scores["rougeL"].fmeasure
            }
        except Exception as e:
            logger.error(f"Error computing ROUGE scores: {e}")
            return defaultdict(float)
    
    def _compute_relevance(self, answer: str, context: str) -> float:
        """
        Compute relevance score between answer and context.
        Uses a combination of lexical overlap and semantic similarity.
        """
        try:
            # Compute word overlap ratio
            context_words = set(word_tokenize(context.lower()))
            answer_words = set(word_tokenize(answer.lower()))
            overlap = len(answer_words.intersection(context_words)) / len(answer_words)
            
            # Adjust score based on answer length ratio
            length_ratio = min(len(answer) / len(context), 1.0)
            
            return (overlap * 0.7 + length_ratio * 0.3)
        except Exception as e:
            logger.error(f"Error computing relevance score: {e}")
            return 0.0
    
    def _evaluate_question_quality(self, question: str) -> float:
        """
        Evaluate the quality of generated question.
        Checks for clarity, specificity, and proper structure.
        """
        try:
            # Initialize score
            score = 1.0
            
            # Check question length (prefer questions between 5 and 25 words)
            words = word_tokenize(question)
            word_count = len(words)
            if word_count < 5:
                score *= 0.7
            elif word_count > 25:
                score *= 0.8
            
            # Check if starts with question word
            question_words = {"what", "who", "where", "when", "why", "how", "which", "whose", "whom"}
            if not any(question.lower().startswith(w) for w in question_words):
                score *= 0.9
            
            # Check for question mark
            if not question.strip().endswith("?"):
                score *= 0.8
            
            return score
            
        except Exception as e:
            logger.error(f"Error evaluating question quality: {e}")
            return 0.0
    
    def _compute_overall_score(self, scores: List[float]) -> float:
        """Compute weighted average of individual scores."""
        weights = [0.2, 0.2, 0.2, 0.2, 0.2]  # Equal weights for now
        return np.average(scores, weights=weights)
    
    def evaluate_diversity(self, qa_pairs: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Evaluate diversity across a set of question-answer pairs.
        
        Args:
            qa_pairs: List of question-answer pairs
            
        Returns:
            Dictionary containing diversity metrics
        """
        if not qa_pairs:
            return defaultdict(float)
            
        try:
            # Calculate question diversity
            questions = [pair["question"] for pair in qa_pairs]
            question_diversity = self._compute_diversity_score(questions)
            
            # Calculate answer diversity
            answers = [pair["answer"] for pair in qa_pairs]
            answer_diversity = self._compute_diversity_score(answers)
            
            return {
                "question_diversity": question_diversity,
                "answer_diversity": answer_diversity,
                "overall_diversity": (question_diversity + answer_diversity) / 2
            }
            
        except Exception as e:
            logger.error(f"Error computing diversity metrics: {e}")
            return defaultdict(float)
    
    def _compute_diversity_score(self, texts: List[str]) -> float:
        """
        Compute diversity score for a list of texts.
        Uses lexical diversity and n-gram overlap measures.
        """
        try:
            # Tokenize all texts
            tokenized = [word_tokenize(text.lower()) for text in texts]
            
            if not tokenized or not tokenized[0]:
                return 0.0
            
            # Calculate lexical diversity
            unique_words = set(word for tokens in tokenized for word in tokens)
            total_words = sum(len(tokens) for tokens in tokenized)
            lexical_diversity = len(unique_words) / total_words
            
            # Calculate average pairwise n-gram overlap
            overlaps = []
            for i in range(len(tokenized)):
                for j in range(i + 1, len(tokenized)):
                    overlap = self._compute_ngram_overlap(tokenized[i], tokenized[j])
                    overlaps.append(overlap)
            
            if not overlaps:
                return lexical_diversity
            
            # Combine metrics (lower overlap is better)
            avg_overlap = np.mean(overlaps)
            diversity_score = (lexical_diversity * 0.5 + (1 - avg_overlap) * 0.5)
            
            return diversity_score
            
        except Exception as e:
            logger.error(f"Error computing diversity score: {e}")
            return 0.0
    
    def _compute_ngram_overlap(self, tokens1: List[str], tokens2: List[str]) -> float:
        """Compute n-gram overlap between two token sequences."""
        def get_ngrams(tokens: List[str], n: int) -> set:
            return set(' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1))
        
        # Use bigrams and trigrams
        bigrams1 = get_ngrams(tokens1, 2)
        bigrams2 = get_ngrams(tokens2, 2)
        trigrams1 = get_ngrams(tokens1, 3)
        trigrams2 = get_ngrams(tokens2, 3)
        
        # Compute Jaccard similarity for both n-gram sizes
        if bigrams1 and bigrams2:
            bigram_overlap = len(bigrams1 & bigrams2) / len(bigrams1 | bigrams2)
        else:
            bigram_overlap = 0
            
        if trigrams1 and trigrams2:
            trigram_overlap = len(trigrams1 & trigrams2) / len(trigrams1 | trigrams2)
        else:
            trigram_overlap = 0
        
        # Return weighted average
        return (bigram_overlap * 0.4 + trigram_overlap * 0.6)