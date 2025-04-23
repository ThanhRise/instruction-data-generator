from typing import Dict, List, Any, Optional
import logging
import spacy
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import re
from collections import defaultdict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class AnswerExtractor:
    """Extracts potential answers from text content."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.extraction_config = config["agent"]["instruction_generation"]["answer_extraction"]
        
        # Initialize models and analyzers
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize NLP components and models."""
        try:
            import spacy
            self.nlp = spacy.load(self.extraction_config.get("spacy_model", "en_core_web_sm"))
            
            # Add special case patterns for handling structured content markers
            ruler = self.nlp.get_pipe("attribute_ruler")
            patterns = [
                {"label": "VISUAL_CONTENT", "pattern": [{"TEXT": {"REGEX": r"\[(OCR Text|Image Caption|Detected Objects|Scene Description|Visual Analysis|Visual Context):"}}]},
                {"label": "CONTENT_MARKER", "pattern": [{"TEXT": {"REGEX": r"===.*==="}}]}
            ]
            for pattern in patterns:
                ruler.add([[pattern]])
                
            # Initialize semantic search model for content similarity
            self.sim_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            
        except Exception as e:
            logger.error(f"Failed to initialize NLP components: {e}")
            raise
    
    def extract_answers(self, content: str) -> List[Dict[str, Any]]:
        """
        Extract potential answers from unified content.
        
        Args:
            content: Text content including both document text and image-derived information
            
        Returns:
            List of extracted answer candidates with metadata
        """
        answers = []
        
        try:
            # Split content into sections based on visual content markers
            sections = self._split_content_sections(content)
            
            for section in sections:
                section_type = section["type"]
                section_text = section["text"]
                
                if section_type == "text":
                    # Process regular text content
                    text_answers = self._extract_text_answers(section_text)
                    answers.extend(text_answers)
                    
                elif section_type == "visual":
                    # Process visual content sections
                    visual_answers = self._extract_visual_answers(section_text)
                    answers.extend(visual_answers)
                    
                elif section_type == "combined":
                    # Process sections with both text and visual content
                    combined_answers = self._extract_combined_answers(section_text)
                    answers.extend(combined_answers)
            
            # Filter and rank answers
            answers = self._filter_answers(answers)
            answers = self._rank_answers(answers)
            
            return answers[:self.extraction_config["max_answers"]]
            
        except Exception as e:
            logger.error(f"Answer extraction failed: {e}")
            return []
    
    def _split_content_sections(self, content: str) -> List[Dict[str, Any]]:
        """Split content into sections based on content type."""
        sections = []
        current_section = {"type": "text", "text": []}
        
        for line in content.split("\n"):
            if line.strip():
                if line.startswith("[") and any(
                    marker in line 
                    for marker in ["OCR Text:", "Image Caption:", "Detected Objects:", 
                                 "Scene Description:", "Visual Analysis:", "Visual Context:"]
                ):
                    # Save previous section if it has content
                    if current_section["text"]:
                        sections.append({
                            "type": current_section["type"],
                            "text": "\n".join(current_section["text"])
                        })
                    
                    # Start new visual section
                    current_section = {"type": "visual", "text": [line]}
                    
                elif line.startswith("==="):
                    # Save previous section if it has content
                    if current_section["text"]:
                        sections.append({
                            "type": current_section["type"],
                            "text": "\n".join(current_section["text"])
                        })
                    
                    # Start new section based on marker
                    if "Visual Content" in line:
                        current_section = {"type": "visual", "text": []}
                    else:
                        current_section = {"type": "text", "text": []}
                        
                else:
                    # If mixing visual and text content, update section type
                    if current_section["type"] == "text" and "[Visual Context:" in line:
                        current_section["type"] = "combined"
                    current_section["text"].append(line)
        
        # Add final section
        if current_section["text"]:
            sections.append({
                "type": current_section["type"],
                "text": "\n".join(current_section["text"])
            })
        
        return sections
    
    def _extract_text_answers(self, text: str) -> List[Dict[str, Any]]:
        """Extract answer candidates from regular text content."""
        answers = []
        doc = self.nlp(text)
        
        # Extract named entities
        if self.extraction_config["extract_entities"]:
            for ent in doc.ents:
                if ent.label_ in self.extraction_config["entity_types"]:
                    answers.append({
                        "text": ent.text,
                        "type": "entity",
                        "subtype": ent.label_,
                        "score": 1.0,
                        "source": "text",
                        "context": text[max(0, ent.start_char - 100):min(len(text), ent.end_char + 100)]
                    })
        
        # Extract key phrases
        if self.extraction_config["extract_phrases"]:
            phrases = self._extract_key_phrases(doc)
            for phrase in phrases:
                answers.append({
                    "text": phrase["text"],
                    "type": "phrase",
                    "subtype": phrase["type"],
                    "score": phrase["score"],
                    "source": "text",
                    "context": phrase["context"]
                })
        
        # Extract factual statements
        if self.extraction_config["extract_facts"]:
            facts = self._extract_facts(doc)
            answers.extend(facts)
        
        return answers
    
    def _extract_visual_answers(self, text: str) -> List[Dict[str, Any]]:
        """Extract answer candidates from visual content sections."""
        answers = []
        
        # Extract OCR text content
        ocr_matches = re.finditer(r"\[OCR Text:([^\]]+)\]", text)
        for match in ocr_matches:
            ocr_text = match.group(1).strip()
            if len(ocr_text) >= self.extraction_config["min_answer_length"]:
                answers.append({
                    "text": ocr_text,
                    "type": "visual",
                    "subtype": "ocr_text",
                    "score": 0.8,  # OCR confidence score
                    "source": "image",
                    "context": text
                })
        
        # Extract image captions
        caption_matches = re.finditer(r"\[Image Caption:([^\]]+)\]", text)
        for match in caption_matches:
            caption = match.group(1).strip()
            answers.append({
                "text": caption,
                "type": "visual",
                "subtype": "caption",
                "score": 0.9,  # Caption confidence
                "source": "image",
                "context": text
            })
        
        # Extract detected objects with high confidence
        object_matches = re.finditer(r"\[Detected Objects:([^\]]+)\]", text)
        for match in object_matches:
            objects_text = match.group(1).strip()
            objects = []
            for obj in objects_text.split(","):
                if "(" in obj and ")" in obj:
                    label, conf = obj.strip().split("(")
                    conf = float(conf.rstrip(")"))
                    if conf > 0.7:  # Only high confidence objects
                        objects.append(label.strip())
            
            if objects:
                answers.append({
                    "text": ", ".join(objects),
                    "type": "visual",
                    "subtype": "objects",
                    "score": 0.85,
                    "source": "image",
                    "context": text
                })
        
        # Extract scene descriptions
        scene_matches = re.finditer(r"\[Scene Description:([^\]]+)\]", text)
        for match in scene_matches:
            description = match.group(1).strip()
            answers.append({
                "text": description,
                "type": "visual",
                "subtype": "scene",
                "score": 0.9,
                "source": "image",
                "context": text
            })
        
        # Extract visual analysis results
        analysis_matches = re.finditer(r"\[Visual Analysis:([^\]]+)\]", text)
        for match in analysis_matches:
            analysis = match.group(1).strip()
            for qa_pair in analysis.split("|"):
                if "->" in qa_pair:
                    q, a = qa_pair.split("->")
                    answers.append({
                        "text": a.strip(),
                        "type": "visual",
                        "subtype": "analysis",
                        "score": 0.85,
                        "source": "image",
                        "context": text,
                        "question": q.strip()
                    })
        
        return answers
    
    def _extract_combined_answers(self, text: str) -> List[Dict[str, Any]]:
        """Extract answers from sections containing both text and visual content."""
        answers = []
        
        # First extract separate answers
        text_answers = self._extract_text_answers(text)
        visual_answers = self._extract_visual_answers(text)
        
        # Add all individual answers
        answers.extend(text_answers)
        answers.extend(visual_answers)
        
        # Look for relationships between text and visual content
        visual_contexts = re.finditer(r"\[Visual Context:([^\]]+)\]", text)
        for context_match in visual_contexts:
            context = context_match.group(1).strip()
            
            # Find nearby text (within 3 paragraphs)
            paragraphs = text.split("\n\n")
            for i, para in enumerate(paragraphs):
                if context in para:
                    start_idx = max(0, i - 3)
                    end_idx = min(len(paragraphs), i + 4)
                    related_text = "\n\n".join(paragraphs[start_idx:end_idx])
                    
                    # Create combined answer if there's a clear relationship
                    related_answers = [
                        ans for ans in visual_answers
                        if ans["context"] in related_text
                    ]
                    
                    for v_ans in related_answers:
                        combined_text = f"{v_ans['text']} (Visual) - Related to: {context}"
                        answers.append({
                            "text": combined_text,
                            "type": "combined",
                            "subtype": f"text_{v_ans['subtype']}",
                            "score": (v_ans['score'] + 0.9) / 2,  # Average with context confidence
                            "source": "text_and_image",
                            "context": related_text
                        })
        
        return answers
    
    def _extract_key_phrases(self, doc) -> List[Dict[str, Any]]:
        """Extract key phrases from text."""
        phrases = []
        
        # Extract noun phrases
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) >= 2:  # At least 2 words
                phrases.append({
                    "text": chunk.text,
                    "type": "noun_phrase",
                    "score": 0.8,
                    "context": doc.text[max(0, chunk.start_char - 100):min(len(doc.text), chunk.end_char + 100)]
                })
        
        # Extract verb phrases
        for token in doc:
            if token.pos_ == "VERB":
                verb_phrase = ""
                for child in token.subtree:
                    verb_phrase += child.text + " "
                if len(verb_phrase.split()) >= 3:  # At least 3 words
                    phrases.append({
                        "text": verb_phrase.strip(),
                        "type": "verb_phrase",
                        "score": 0.75,
                        "context": doc.text[max(0, token.idx - 100):min(len(doc.text), token.idx + len(verb_phrase) + 100)]
                    })
        
        return phrases
    
    def _extract_facts(self, doc) -> List[Dict[str, Any]]:
        """Extract factual statements from text."""
        facts = []
        
        for sent in doc.sents:
            # Look for factual indicators
            if any(token.dep_ in {"ROOT", "nsubj"} for token in sent):
                fact_score = self._calculate_fact_score(sent)
                if fact_score >= self.extraction_config["min_fact_score"]:
                    facts.append({
                        "text": sent.text,
                        "type": "fact",
                        "subtype": "statement",
                        "score": fact_score,
                        "source": "text",
                        "context": doc.text[max(0, sent.start_char - 100):min(len(doc.text), sent.end_char + 100)]
                    })
        
        return facts
    
    def _calculate_fact_score(self, sent) -> float:
        """Calculate confidence score for a potential fact."""
        score = 0.5  # Base score
        
        # Increase score based on indicators
        if any(token.pos_ == "NUM" for token in sent):
            score += 0.1  # Contains numbers
        if any(token.ent_type_ in {"DATE", "TIME", "PERCENT", "MONEY"} for token in sent):
            score += 0.1  # Contains specific entities
        if any(token.dep_ in {"nsubj", "dobj"} for token in sent):
            score += 0.1  # Has clear subject-object structure
        if any(token.pos_ == "VERB" and token.tag_ in {"VBD", "VBZ"} for token in sent):
            score += 0.1  # Uses definitive verb tenses
        
        return min(1.0, score)
    
    def _filter_answers(self, answers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter answer candidates based on quality criteria."""
        filtered = []
        
        for answer in answers:
            # Check minimum length
            if len(answer["text"].split()) < self.extraction_config["min_answer_length"]:
                continue
                
            # Check maximum length
            if len(answer["text"].split()) > self.extraction_config["max_answer_length"]:
                continue
            
            # Check minimum score
            if answer["score"] < self.extraction_config["min_answer_score"]:
                continue
            
            # Remove duplicates using semantic similarity
            if not self._is_duplicate(answer, filtered):
                filtered.append(answer)
        
        return filtered
    
    def _rank_answers(self, answers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank answers by relevance and quality."""
        if not answers:
            return []
            
        # Calculate ranking scores
        for answer in answers:
            ranking_score = answer["score"]  # Start with confidence score
            
            # Adjust based on answer type
            type_weights = {
                "combined": 1.2,    # Prefer combined text-visual answers
                "fact": 1.1,       # Prefer factual statements
                "visual": 1.0,     # Standard weight for visual content
                "entity": 0.9,     # Slightly lower weight for single entities
                "phrase": 0.8      # Lower weight for general phrases
            }
            ranking_score *= type_weights.get(answer["type"], 1.0)
            
            # Adjust based on length (prefer medium-length answers)
            length = len(answer["text"].split())
            if 5 <= length <= 15:
                ranking_score *= 1.1
            
            answer["ranking_score"] = ranking_score
        
        # Sort by ranking score
        return sorted(answers, key=lambda x: x["ranking_score"], reverse=True)
    
    def _is_duplicate(self, candidate: Dict[str, Any], existing: List[Dict[str, Any]]) -> bool:
        """Check if an answer is semantically similar to existing answers."""
        if not existing:
            return False
            
        candidate_embedding = self.sim_model.encode([candidate["text"]])[0]
        
        for answer in existing:
            existing_embedding = self.sim_model.encode([answer["text"]])[0]
            similarity = cosine_similarity(
                candidate_embedding.reshape(1, -1),
                existing_embedding.reshape(1, -1)
            )[0][0]
            
            if similarity > self.extraction_config["duplicate_threshold"]:
                return True
        
        return False