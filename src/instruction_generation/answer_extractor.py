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
from ..models.model_loader import ModelLoader

logger = logging.getLogger(__name__)

class AnswerExtractor:
    """Extracts potential answers from text content."""
    
    def __init__(self, config: Dict[str, Any], model_name: Optional[str] = None):
        self.config = config
        self.extraction_config = config["agent"]["instruction_generation"]["answer_extraction"]
        
        # Initialize models and analyzers
        self._initialize_components()
        
        # Get shared LLM instance
        self.model_loader = ModelLoader(config)
        if model_name:
            self.model_name = model_name
        else:
            self.model_name = self.config["models"]["llm_models"].get("default", "gpt-4")
        
        self.llm = self.model_loader.get_shared_model("answer_extraction", self.model_name)
        
        # Load few-shot examples for answer extraction
        self.few_shot_examples = self._load_few_shot_examples()

    def _initialize_components(self):
        """Initialize NLP components and models."""
        try:
            import spacy
            self.nlp = spacy.load(self.extraction_config.get("spacy_model", "en_core_web_sm"))
            
            # Add special case patterns for handling structured content markers
            ruler = self.nlp.get_pipe("attribute_ruler")
            patterns = [
                {
                    "patterns": [[{"TEXT": {"REGEX": r"\[(OCR Text|Image Caption|Detected Objects|Scene Description|Visual Analysis|Visual Context):"}}]], 
                    "attrs": {"ENT_TYPE": "VISUAL_CONTENT"}
                },
                {
                    "patterns": [[{"TEXT": {"REGEX": r"===.*==="}}]], 
                    "attrs": {"ENT_TYPE": "CONTENT_MARKER"}
                }
            ]
            for pattern in patterns:
                ruler.add(pattern["patterns"], pattern["attrs"])
                
            # Initialize semantic search model for content similarity
            self.sim_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            
        except Exception as e:
            logger.error(f"Failed to initialize NLP components: {e}")
            raise
    
    def _load_few_shot_examples(self) -> str:
        """Load few-shot examples for answer extraction."""
        examples = """Text: The Tesla Model S was first introduced in 2012. It has a range of up to 405 miles and can accelerate from 0-60 mph in 2.3 seconds.
Answers:
1. 2012 (Year of Tesla Model S introduction)
2. 405 miles (Range of Tesla Model S)
3. 2.3 seconds (0-60 mph acceleration time)

Text: Python was created by Guido van Rossum and released in 1991. It emphasizes code readability with its notable use of significant indentation.
Answers:
1. Guido van Rossum (Creator of Python)
2. 1991 (Release year)
3. code readability (Key emphasis of Python)
4. significant indentation (Notable feature)

This shows how to identify key pieces of information that could serve as answers."""
        return examples

    def _extract_llm_answers(self, text: str) -> List[Dict[str, Any]]:
        """Extract answers using LLM with chain-of-thought prompting."""
        try:
            # Prepare prompt with few-shot examples and chain-of-thought
            prompt = f"""You are an expert at identifying key information in text that could serve as answers to questions.
            
{self.few_shot_examples}

Now, analyze this text and identify key pieces of information that could serve as answers.
Use chain-of-thought reasoning to explain why each piece of information is important.

Text: {text}

Think through the following steps:
1. Identify factual information (dates, numbers, names, etc.)
2. Look for key concepts and definitions
3. Consider important relationships or cause-effect pairs
4. Note any significant descriptions or characteristics

Then list the answers in this format:
Answer: [piece of information]
Reasoning: [why this is important information]
Confidence: [score between 0-1]
Type: [entity/fact/description]

Begin your analysis:"""

            # Get LLM response
            model_dict = self.llm
            if "vllm" in str(type(model_dict["model"])):
                response = self._get_vllm_response(model_dict["model"], prompt)
            else:
                response = model_dict["model"](prompt)

            # Parse LLM output into answer dictionaries
            answers = []
            current_answer = {}
            
            for line in response.split('\n'):
                line = line.strip()
                if not line:
                    if current_answer.get('text'):
                        answers.append(current_answer.copy())
                        current_answer = {}
                    continue
                    
                if line.startswith('Answer:'):
                    if current_answer.get('text'):
                        answers.append(current_answer.copy())
                    current_answer = {
                        'text': line[7:].strip(),
                        'source': 'llm',
                        'context': text
                    }
                elif line.startswith('Reasoning:'):
                    current_answer['reasoning'] = line[10:].strip()
                elif line.startswith('Confidence:'):
                    try:
                        current_answer['score'] = float(line[11:].strip())
                    except:
                        current_answer['score'] = 0.8
                elif line.startswith('Type:'):
                    current_answer['type'] = line[5:].strip().lower()
                    current_answer['subtype'] = current_answer['type']

            # Add final answer if exists
            if current_answer.get('text'):
                answers.append(current_answer)

            # Verify answers are present in text
            verified_answers = []
            for answer in answers:
                if self._verify_answer_in_text(answer['text'], text):
                    verified_answers.append(answer)
                else:
                    logger.warning(f"Answer '{answer['text']}' not found in source text - discarding")

            return verified_answers

        except Exception as e:
            logger.error(f"Error in LLM answer extraction: {e}")
            return []

    def _verify_answer_in_text(self, answer: str, text: str) -> bool:
        """Verify that an answer is present in the source text."""
        # Clean and normalize text for comparison
        def normalize_text(t):
            return re.sub(r'\s+', ' ', t.lower().strip())
        
        norm_answer = normalize_text(answer)
        norm_text = normalize_text(text)
        
        # Direct match
        if norm_answer in norm_text:
            return True
            
        # Check for semantic similarity using sentence embeddings
        answer_embedding = self.sim_model.encode([norm_answer])[0]
        
        # Split text into chunks around the same length as the answer
        words = norm_text.split()
        chunk_size = len(norm_answer.split())
        
        for i in range(len(words) - chunk_size + 1):
            chunk = " ".join(words[i:i+chunk_size])
            chunk_embedding = self.sim_model.encode([chunk])[0]
            
            similarity = cosine_similarity(
                answer_embedding.reshape(1, -1),
                chunk_embedding.reshape(1, -1)
            )[0][0]
            
            if similarity > 0.8:  # High semantic similarity threshold
                return True
                
        return False

    def extract_answers(self, content: str) -> List[Dict[str, Any]]:
        """Extract potential answers from unified content using multiple methods."""
        all_answers = []
        
        try:
            # Get traditional NER/rule-based answers
            traditional_answers = self._extract_text_answers(content)
            all_answers.extend(traditional_answers)
            
            # Get LLM-generated answers
            llm_answers = self._extract_llm_answers(content)
            all_answers.extend(llm_answers)
            
            # Process any visual content sections
            visual_answers = self._extract_visual_answers(content)
            all_answers.extend(visual_answers)
            
            # Process combined content sections
            combined_answers = self._extract_combined_answers(content)
            all_answers.extend(combined_answers)
            
            # Filter and rank all answers
            filtered_answers = self._filter_answers(all_answers)
            ranked_answers = self._rank_answers(filtered_answers)
            
            return ranked_answers[:self.extraction_config["max_answers"]]
            
        except Exception as e:
            logger.error(f"Error in answer extraction: {e}")
            return []

    def _get_vllm_response(self, model: Any, prompt: str) -> str:
        """Get response from vLLM model."""
        outputs = model.generate([prompt], sampling_params=self.llm["sampling_params"])
        return outputs[0].outputs[0].text if outputs else ""

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