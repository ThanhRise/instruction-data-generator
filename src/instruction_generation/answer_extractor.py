from typing import Dict, List, Any, Optional
import logging
import spacy
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import re
from collections import defaultdict
from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

class AnswerExtractor:
    """Extracts potential answers from text and image-related content."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the answer extractor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Load NLP models
        try:
            self.nlp = spacy.load("en_core_web_lg")
        except Exception as e:
            logger.error(f"Error loading spaCy model: {e}")
            self.nlp = None
            
        try:
            self.ner_tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
            self.ner_model = AutoModelForTokenClassification.from_pretrained(
                "dslim/bert-base-NER",
                torch_dtype=torch.float16,
                device_map="auto"
            )
        except Exception as e:
            logger.error(f"Error loading NER model: {e}")
            self.ner_tokenizer = None
            self.ner_model = None
        
        # Text splitter for chunking long texts
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", ", ", " ", ""]
        )
    
    def extract_answers(
        self,
        content: Dict[str, Any],
        context_window: int = 200
    ) -> List[Dict[str, Any]]:
        """
        Extract potential answers from content, considering both text and related images.
        
        Args:
            content: Dictionary containing text, related images, and document context
            context_window: Number of characters to include as context around answers
            
        Returns:
            List of extracted answers with their contexts and types
        """
        answers = []
        
        try:
            text_item = content.get("text", {})
            text_content = text_item.get("content", "")
            text_type = text_item.get("type", "text")
            
            # Extract answers from main text
            text_answers = self._extract_from_text(text_content)
            
            # Add document context to answers
            doc_context = content.get("document_context")
            if doc_context:
                for answer in text_answers:
                    answer["document_type"] = doc_context["type"]
                    if "page" in text_item:
                        answer["page"] = text_item["page"]
                    elif "slide" in text_item:
                        answer["slide"] = text_item["slide"]
                    elif "sheet" in text_item:
                        answer["sheet"] = text_item["sheet"]
            
            answers.extend(text_answers)
            
            # Process related images and their captions/context
            related_images = content.get("related_images", [])
            for img_data in related_images:
                image = img_data.get("image", {})
                rel_type = img_data.get("relationship_type")
                
                # Extract answers that combine text and image information
                combined_answers = self._extract_from_image_context(
                    text_content,
                    image,
                    rel_type,
                    doc_context
                )
                answers.extend(combined_answers)
            
            # Add source and location context
            for answer in answers:
                answer["source"] = text_item.get("source", "unknown")
                if "context" not in answer:
                    answer["context"] = self._get_context(
                        text_content,
                        answer["answer"],
                        context_window
                    )
            
            return answers
            
        except Exception as e:
            logger.error(f"Error extracting answers: {e}")
            return []
    
    def _extract_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract answers from text content."""
        answers = []
        
        try:
            if not text.strip():
                return answers
                
            # Extract named entities
            if self.ner_model and self.ner_tokenizer:
                entities = self._extract_entities(text)
                answers.extend(entities)
            
            # Extract key phrases and sentences using spaCy
            if self.nlp:
                doc = self.nlp(text)
                
                # Extract noun phrases
                for chunk in doc.noun_chunks:
                    if len(chunk.text.split()) > 1:  # Multi-word phrases only
                        answers.append({
                            "answer": chunk.text,
                            "type": "noun_phrase",
                            "confidence": 0.7
                        })
                
                # Extract key sentences
                for sent in doc.sents:
                    # Filter important sentences (containing entities or key information)
                    if (len(sent.ents) > 0 or
                        any(token.pos_ in ["VERB", "NUM"] for token in sent) and
                        len(sent.text.split()) >= 5):
                        answers.append({
                            "answer": sent.text,
                            "type": "key_sentence",
                            "confidence": 0.8
                        })
                
                # Extract numerical facts
                number_patterns = self._extract_numerical_facts(doc)
                answers.extend(number_patterns)
            
            return answers
            
        except Exception as e:
            logger.error(f"Error in text extraction: {e}")
            return []
    
    def _extract_from_image_context(
        self,
        text: str,
        image: Dict[str, Any],
        relationship_type: str,
        doc_context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Extract answers that combine image and text information."""
        answers = []
        
        try:
            # Extract image-specific information
            image_type = image.get("type", "")
            
            # Handle different types of image-text relationships
            if relationship_type == "page_content":
                # For PDF pages, focus on visual elements mentioned in text
                answers.extend(self._extract_visual_references(text))
                
            elif relationship_type == "slide_content":
                # For PowerPoint slides, extract bullet points and visual descriptions
                answers.extend(self._extract_slide_content(text))
                
            elif relationship_type == "embedded_content":
                # For embedded images, find direct references to the image
                answers.extend(self._extract_image_references(text))
            
            # Add image context to answers
            for answer in answers:
                answer["has_visual_context"] = True
                answer["image_type"] = image_type
                if doc_context:
                    answer["document_type"] = doc_context["type"]
            
            return answers
            
        except Exception as e:
            logger.error(f"Error in image-context extraction: {e}")
            return []
    
    def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities using BERT-NER."""
        entities = []
        
        try:
            # Tokenize and get predictions
            inputs = self.ner_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.ner_model.device)
            
            with torch.no_grad():
                outputs = self.ner_model(**inputs)
            
            # Process predictions
            predictions = outputs.logits.argmax(-1)[0].tolist()
            tokens = self.ner_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            
            current_entity = {"text": "", "type": "", "confidence": 0.0}
            
            for token, pred in zip(tokens, predictions):
                # Convert prediction ID to label
                label = self.ner_model.config.id2label[pred]
                
                if label.startswith("B-"):
                    # Start of new entity
                    if current_entity["text"]:
                        entities.append({
                            "answer": current_entity["text"].strip(),
                            "type": "named_entity",
                            "entity_type": current_entity["type"],
                            "confidence": current_entity["confidence"]
                        })
                    
                    current_entity = {
                        "text": token.replace("#", ""),
                        "type": label[2:],
                        "confidence": 0.9
                    }
                    
                elif label.startswith("I-") and current_entity["text"]:
                    # Inside an entity
                    current_entity["text"] += " " + token.replace("#", "")
                    
                elif current_entity["text"]:
                    # End of entity
                    entities.append({
                        "answer": current_entity["text"].strip(),
                        "type": "named_entity",
                        "entity_type": current_entity["type"],
                        "confidence": current_entity["confidence"]
                    })
                    current_entity = {"text": "", "type": "", "confidence": 0.0}
            
            # Add last entity if exists
            if current_entity["text"]:
                entities.append({
                    "answer": current_entity["text"].strip(),
                    "type": "named_entity",
                    "entity_type": current_entity["type"],
                    "confidence": current_entity["confidence"]
                })
            
            return entities
            
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return []
    
    def _extract_numerical_facts(self, doc) -> List[Dict[str, Any]]:
        """Extract numerical facts and patterns."""
        facts = []
        
        for sent in doc.sents:
            num_tokens = [token for token in sent if token.like_num]
            
            if num_tokens:
                # Find context around numbers
                for num in num_tokens:
                    # Look for measurement patterns
                    if num.i + 1 < len(sent) and sent[num.i + 1].text.lower() in {
                        "kg", "km", "meters", "years", "dollars", "percent", "%"
                    }:
                        facts.append({
                            "answer": sent.text,
                            "type": "measurement",
                            "confidence": 0.85
                        })
                    
                    # Look for date patterns
                    elif any(date_token.ent_type_ == "DATE" for date_token in sent):
                        facts.append({
                            "answer": sent.text,
                            "type": "date_fact",
                            "confidence": 0.85
                        })
                    
                    # Look for statistical statements
                    elif any(token.text.lower() in {
                        "average", "mean", "median", "total", "approximately",
                        "about", "roughly", "estimated"
                    } for token in sent):
                        facts.append({
                            "answer": sent.text,
                            "type": "statistic",
                            "confidence": 0.8
                        })
        
        return facts
    
    def _extract_visual_references(self, text: str) -> List[Dict[str, Any]]:
        """Extract references to visual elements in text."""
        visual_patterns = []
        
        # Pattern for visual references
        patterns = [
            (r"(?:In |The |This )(?:figure|image|picture|illustration|photo|graph|chart|diagram)[^.]*\.", "visual_reference"),
            (r"(?:shows|displays|depicts|represents|visualizes)[^.]*\.", "visual_description"),
            (r"(?:as shown|as illustrated|as depicted|as demonstrated)[^.]*\.", "visual_reference"),
            (r"(?:can be seen|is visible|appears|looking at)[^.]*\.", "visual_observation")
        ]
        
        for pattern, ref_type in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                visual_patterns.append({
                    "answer": match.group(0),
                    "type": ref_type,
                    "confidence": 0.85
                })
        
        return visual_patterns
    
    def _extract_slide_content(self, text: str) -> List[Dict[str, Any]]:
        """Extract content from presentation slides."""
        slide_content = []
        
        # Extract bullet points
        bullet_points = re.findall(r'(?:^|\n)[•\-\*]\s*([^\n]+)', text)
        for point in bullet_points:
            if len(point.split()) >= 3:  # Minimum length for meaningful content
                slide_content.append({
                    "answer": point,
                    "type": "bullet_point",
                    "confidence": 0.9
                })
        
        # Extract title-like statements
        title_patterns = re.findall(r'(?:^|\n)([A-Z][^.!?\n]{15,100}[.!?])', text)
        for title in title_patterns:
            slide_content.append({
                "answer": title,
                "type": "slide_title",
                "confidence": 0.85
            })
        
        return slide_content
    
    def _extract_image_references(self, text: str) -> List[Dict[str, Any]]:
        """Extract references to embedded images."""
        references = []
        
        # Find sentences referring to embedded images
        doc = self.nlp(text)
        for sent in doc.sents:
            lower_sent = sent.text.lower()
            
            # Check for image references
            if any(term in lower_sent for term in [
                "image", "figure", "photo", "picture", "illustration",
                "shown", "depicted", "illustrated"
            ]):
                references.append({
                    "answer": sent.text,
                    "type": "image_reference",
                    "confidence": 0.8
                })
        
        return references
    
    def _get_context(self, text: str, answer: str, window: int) -> str:
        """Get surrounding context for an answer."""
        try:
            # Find the answer position
            answer_pos = text.lower().find(answer.lower())
            if answer_pos == -1:
                return answer
            
            # Get context window
            start = max(0, answer_pos - window)
            end = min(len(text), answer_pos + len(answer) + window)
            
            # Expand to sentence boundaries
            while start > 0 and text[start] not in ".!?\n":
                start -= 1
            while end < len(text) and text[end] not in ".!?\n":
                end += 1
            
            return text[start:end].strip()
            
        except Exception as e:
            logger.error(f"Error getting context: {e}")
            return answer