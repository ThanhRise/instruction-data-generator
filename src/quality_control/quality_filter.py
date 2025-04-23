from typing import Dict, List, Any
import logging
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re

logger = logging.getLogger(__name__)

class QualityFilter:
    """Filters and validates generated instruction data."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize quality filter with configuration."""
        self.config = config
        self.quality_config = config["agent"]["quality_control"]
        
        # Load quality metrics
        self.metrics = self._load_metrics()
        
        # Initialize sentence transformer for semantic similarity
        self.sim_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    def filter_qa_pairs(
        self,
        qa_pairs: List[Dict[str, Any]],
        contexts: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        """Filter QA pairs based on quality criteria."""
        filtered_pairs = []
        
        for pair in qa_pairs:
            if self._meets_quality_criteria(pair, contexts[pair["source"]]):
                filtered_pairs.append(pair)
        
        return filtered_pairs
    
    def validate_against_source(
        self,
        qa_pairs: List[Dict[str, Any]],
        contexts: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        """Validate QA pairs against source content."""
        validated_pairs = []
        
        for pair in qa_pairs:
            source_content = contexts[pair["source"]]
            
            # Extract all text content including image-derived text
            text_content = self._extract_all_text(source_content)
            
            # Check if answer is supported by source content
            if self._validate_answer(pair["answer"], text_content):
                # Check for answer presence in different content types
                answer_locations = self._locate_answer(pair["answer"], source_content)
                pair["metadata"] = {
                    **pair.get("metadata", {}),
                    "answer_locations": answer_locations
                }
                validated_pairs.append(pair)
        
        return validated_pairs
    
    def _extract_all_text(self, content: str) -> str:
        """Extract all text content including OCR and image captions."""
        # Split content into sections
        sections = content.split("\n\n")
        extracted_text = []
        
        for section in sections:
            # Keep original text
            if not (section.startswith("[OCR Text:") or section.startswith("[Image Caption:")):
                extracted_text.append(section)
            # Extract OCR text
            ocr_match = re.search(r"\[OCR Text: (.*?)\]", section)
            if ocr_match:
                extracted_text.append(ocr_match.group(1))
            # Extract image captions
            caption_match = re.search(r"\[Image Caption: (.*?)\]", section)
            if caption_match:
                extracted_text.append(caption_match.group(1))
        
        return "\n\n".join(extracted_text)
    
    def _locate_answer(self, answer: str, content: str) -> Dict[str, bool]:
        """Locate where the answer appears in different content types."""
        locations = {
            "main_text": False,
            "ocr_text": False,
            "image_caption": False
        }
        
        # Check main text (excluding OCR and captions)
        main_text = "\n\n".join(
            section for section in content.split("\n\n")
            if not (section.startswith("[OCR Text:") or section.startswith("[Image Caption:"))
        )
        locations["main_text"] = self._text_similar(answer, main_text)
        
        # Check OCR text
        ocr_sections = [
            re.search(r"\[OCR Text: (.*?)\]", section).group(1)
            for section in content.split("\n\n")
            if section.startswith("[OCR Text:")
        ]
        if ocr_sections:
            locations["ocr_text"] = any(
                self._text_similar(answer, ocr_text)
                for ocr_text in ocr_sections
            )
        
        # Check image captions
        caption_sections = [
            re.search(r"\[Image Caption: (.*?)\]", section).group(1)
            for section in content.split("\n\n")
            if section.startswith("[Image Caption:")
        ]
        if caption_sections:
            locations["image_caption"] = any(
                self._text_similar(answer, caption)
                for caption in caption_sections
            )
        
        return locations
    
    def _meets_quality_criteria(
        self,
        pair: Dict[str, Any],
        context: str
    ) -> bool:
        """Check if QA pair meets quality criteria."""
        # Get thresholds from config
        thresholds = self.quality_config["thresholds"]
        
        # Basic length checks
        if not self._check_length_requirements(pair):
            return False
        
        # Check answer presence in context
        if not self._validate_answer(pair["answer"], context):
            return False
        
        # Check relevance score
        if not self._check_relevance(pair["question"], pair["answer"], context):
            return False
        
        # Apply content filters
        if not self._apply_content_filters(pair):
            return False
        
        return True
    
    def _check_length_requirements(self, pair: Dict[str, Any]) -> bool:
        """Check if QA pair meets length requirements."""
        gen_config = self.config["agent"]["instruction_generation"]
        
        question_len = len(pair["question"].split())
        answer_len = len(pair["answer"].split())
        
        return (
            gen_config["min_question_length"] <= question_len <= gen_config["max_question_length"]
            and gen_config["min_answer_length"] <= answer_len <= gen_config["max_answer_length"]
        )
    
    def _validate_answer(self, answer: str, context: str) -> bool:
        """Validate if answer is supported by context."""
        # Use sentence embeddings to check semantic similarity
        context_embedding = self.sim_model.encode([context])
        answer_embedding = self.sim_model.encode([answer])
        
        similarity = cosine_similarity(context_embedding, answer_embedding)[0][0]
        return similarity >= self.quality_config["thresholds"]["answer_presence"]
    
    def _check_relevance(
        self,
        question: str,
        answer: str,
        context: str
    ) -> bool:
        """Check relevance of QA pair to context."""
        # Encode texts
        encodings = self.sim_model.encode([question, answer, context])
        
        # Calculate similarities
        q_c_sim = cosine_similarity([encodings[0]], [encodings[2]])[0][0]
        a_c_sim = cosine_similarity([encodings[1]], [encodings[2]])[0][0]
        
        # Get threshold
        threshold = self.quality_config["thresholds"]["relevance"]
        
        return q_c_sim >= threshold and a_c_sim >= threshold
    
    def _apply_content_filters(self, pair: Dict[str, Any]) -> bool:
        """Apply content filters to QA pair."""
        filters = self.quality_config["content_filters"]
        
        for filter_type in filters:
            if not self._check_filter(filter_type, pair):
                return False
        
        return True
    
    def _check_filter(
        self,
        filter_type: str,
        pair: Dict[str, Any]
    ) -> bool:
        """Check specific content filter."""
        text = f"{pair['question']} {pair['answer']}"
        
        if filter_type == "profanity":
            return not self._contains_profanity(text)
        elif filter_type == "personal_info":
            return not self._contains_personal_info(text)
        elif filter_type == "code_snippets":
            return not self._contains_code(text)
        
        return True
    
    def _text_similar(self, text1: str, text2: str) -> bool:
        """Check if two texts are semantically similar."""
        if not text1 or not text2:
            return False
            
        # Use sentence embeddings for similarity
        embeddings = self.sim_model.encode([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        
        return similarity >= self.quality_config["thresholds"]["relevance"]
    
    def _load_metrics(self) -> Dict[str, Any]:
        """Load quality metrics from configuration."""
        metrics = {}
        
        for metric in self.quality_config["metrics"]:
            if metric == "rouge":
                from rouge_score import rouge_scorer
                metrics["rouge"] = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            elif metric == "bert_score":
                from bert_score import BERTScorer
                metrics["bert_score"] = BERTScorer(lang="en", rescale_with_baseline=True)
        
        return metrics