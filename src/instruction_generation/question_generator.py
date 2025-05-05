from typing import Dict, List, Any, Optional
import logging
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

from ..models.model_loader import ModelLoader  # Assuming ModelLoader is defined in model_loader.py

logger = logging.getLogger(__name__)

class QuestionGenerator:
    """Generates questions from extracted answers."""
    
    def __init__(self, config: Dict[str, Any], model_name: Optional[str] = None):
        """Initialize the question generator."""
        self.config = config
        self.gen_config = config["agent"]["instruction_generation"]
        
        # Initialize NLP components first
        self._initialize_components()
        
        # Get shared LLM instance
        self.model_loader = ModelLoader(config)
        self.model_name = model_name if model_name else self.config["models"]["llm_models"].get("default", "gpt-4")
        self.llm = self.model_loader.get_shared_llm(self.model_name)
        
        # Initialize other components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize NLP components and generation model."""
        try:
            import spacy
            self.nlp = spacy.load(self.gen_config.get("spacy_model", "en_core_web_sm"))
            
            # Initialize semantic search model
            from sentence_transformers import SentenceTransformer
            self.sim_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            
            # Load question templates
            self.templates = self._load_question_templates()
            
            # Initialize question type classifier if enabled
            if self.gen_config.get("use_question_classifier", True):
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
                self.q_classifier = AutoModelForSequenceClassification.from_pretrained(
                    "carrassi-ni/bert-base-trec-question-classification"
                )
                self.q_tokenizer = AutoTokenizer.from_pretrained(
                    "carrassi-ni/bert-base-trec-question-classification"
                )
                
        except Exception as e:
            logger.error(f"Failed to initialize question generation components: {e}")
            raise
    
    def _load_question_templates(self) -> Dict[str, List[str]]:
        """Load question templates for different answer types."""
        templates = {
            "text": {
                "entity": [
                    "What is {placeholder}?",
                    "Can you describe {placeholder}?",
                    "What do you know about {placeholder}?"
                ],
                "fact": [
                    "What is true about {subject}?",
                    "What happened with {subject}?",
                    "Could you explain how {subject} relates to this?"
                ],
                "phrase": [
                    "What does {placeholder} refer to?",
                    "What is the significance of {placeholder}?",
                    "How would you explain {placeholder}?"
                ]
            },
            "visual": {
                "ocr_text": [
                    "What text appears in this image?",
                    "What written content is shown?",
                    "What text can be read from this image?"
                ],
                "caption": [
                    "What does this image show?",
                    "What is depicted in this image?",
                    "Could you describe what this image contains?"
                ],
                "objects": [
                    "What objects can be identified in this image?",
                    "What items are visible in this scene?",
                    "What are the main elements present in this image?"
                ],
                "scene": [
                    "What is happening in this scene?",
                    "What type of scene is shown?",
                    "What is the setting or context of this image?"
                ],
                "analysis": [
                    "What can be observed in this image?",
                    "What visual elements are noteworthy?",
                    "What details stand out in this image?"
                ]
            },
            "combined": {
                "text_ocr": [
                    "How does the text in the image relate to {context}?",
                    "What connection exists between the image text and {context}?",
                    "How does the written content support {context}?"
                ],
                "text_caption": [
                    "How does this image illustrate {context}?",
                    "What visual evidence supports {context}?",
                    "How does the image content relate to {context}?"
                ],
                "text_objects": [
                    "Which objects in the image are relevant to {context}?",
                    "How do the visible elements connect to {context}?",
                    "What items in the image support {context}?"
                ],
                "text_scene": [
                    "How does this scene relate to {context}?",
                    "What aspects of the scene illustrate {context}?",
                    "How does the visual setting connect to {context}?"
                ]
            }
        }
        
        # Add dynamic templates based on configuration
        if self.gen_config.get("custom_templates"):
            for category, type_templates in self.gen_config["custom_templates"].items():
                if category not in templates:
                    templates[category] = {}
                templates[category].update(type_templates)
        
        return templates
    
    def generate_questions(self, content: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate questions from content and extracted answers.
        
        Args:
            content: Dictionary containing source content and extracted answers
            
        Returns:
            List of generated question-answer pairs
        """
        qa_pairs = []
        
        try:
            source = content["source"]
            text_content = content["content"]
            answers = content["extracted_answers"]
            
            # Group answers by type
            answers_by_type = self._group_answers(answers)
            
            # Generate questions for each answer type
            for ans_type, type_answers in answers_by_type.items():
                category = ans_type.split("_")[0]  # text, visual, or combined
                
                if category == "text":
                    pairs = self._generate_text_questions(type_answers, text_content)
                elif category == "visual":
                    pairs = self._generate_visual_questions(type_answers, text_content)
                else:  # combined
                    pairs = self._generate_combined_questions(type_answers, text_content)
                
                qa_pairs.extend(pairs)
            
            # Filter and rank question-answer pairs
            qa_pairs = self._filter_qa_pairs(qa_pairs)
            qa_pairs = self._rank_qa_pairs(qa_pairs)
            
            # Add metadata
            for pair in qa_pairs:
                pair["source"] = source
                pair["confidence"] = self._calculate_confidence(pair)
            
            return qa_pairs[:self.gen_config["max_questions"]]
            
        except Exception as e:
            logger.error(f"Question generation failed: {e}")
            return []
    
    def _group_answers(self, answers: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group answers by their type and subtype."""
        grouped = defaultdict(list)
        for answer in answers:
            key = f"{answer['type']}_{answer['subtype']}"
            grouped[key].append(answer)
        return grouped
    
    def _generate_text_questions(
        self,
        answers: List[Dict[str, Any]],
        content: str
    ) -> List[Dict[str, Any]]:
        """Generate questions for text-based answers using LLM and traditional methods."""
        qa_pairs = []
        
        for answer in answers:
            # Get base questions using templates first
            template_questions = self._generate_from_templates(answer, content)
            qa_pairs.extend(template_questions)
            
            # Generate questions using LLM
            llm_questions = self._generate_llm_questions(answer, content)
            qa_pairs.extend(llm_questions)
            
            # Refine questions through multi-turn generation
            refined_pairs = self._refine_questions(qa_pairs, answer, content)
            
            qa_pairs = refined_pairs
        
        return qa_pairs
    
    def _generate_llm_questions(self, answer: Dict[str, Any], context: str) -> List[Dict[str, Any]]:
        """Generate questions using LLM with various question types."""
        try:
            # Prepare prompt for diverse question generation
            prompt = f"""You are an expert at generating diverse and high-quality questions based on given answers.

Context: {context}

Answer: {answer['text']}
Answer Type: {answer['type']}
Reasoning: {answer.get('reasoning', 'Key information from the text')}

Generate 3-5 different types of questions that would lead to this answer. Include:
1. A factual/direct question
2. An inferential/reasoning question
3. A descriptive/detail-oriented question

For each question:
1. Ensure it can be answered accurately using ONLY the provided context
2. Make it clear and specific
3. Use varied question structures
4. Include verification that answer matches the question

Format each question as:
Question Type: [type]
Question: [question]
Verification: [explain how this matches the answer]
Complexity: [score 0-1]

Begin generation:"""

            # Get LLM response
            model_dict = self.llm
            if "vllm" in str(type(model_dict["model"])):
                response = self._get_vllm_response(model_dict["model"], prompt)
            else:
                response = model_dict["model"](prompt)

            # Parse response into question dictionaries
            questions = []
            current_question = {}
            
            for line in response.split('\n'):
                line = line.strip()
                if not line:
                    if current_question.get('question'):
                        questions.append(current_question.copy())
                        current_question = {}
                    continue
                
                if line.startswith('Question Type:'):
                    if current_question.get('question'):
                        questions.append(current_question.copy())
                    current_question = {
                        'type': line[13:].strip().lower(),
                        'answer': answer['text'],
                        'context': context,
                        'source': 'llm'
                    }
                elif line.startswith('Question:'):
                    current_question['question'] = line[9:].strip()
                elif line.startswith('Verification:'):
                    current_question['verification'] = line[13:].strip()
                elif line.startswith('Complexity:'):
                    try:
                        current_question['score'] = float(line[11:].strip())
                    except:
                        current_question['score'] = 0.8

            # Add final question if exists
            if current_question.get('question'):
                questions.append(current_question)

            # Validate questions
            validated_questions = []
            for q in questions:
                if self._validate_question(q['question'], q['answer'], context):
                    validated_questions.append(q)
                else:
                    logger.warning(f"Question '{q['question']}' failed validation - discarding")

            return validated_questions

        except Exception as e:
            logger.error(f"Error in LLM question generation: {e}")
            return []

    def _refine_questions(
        self,
        questions: List[Dict[str, Any]],
        answer: Dict[str, Any],
        context: str
    ) -> List[Dict[str, Any]]:
        """Refine questions through multi-turn generation."""
        try:
            # Prepare refinement prompt
            questions_text = "\n".join([
                f"Q{i+1}: {q['question']}\nType: {q['type']}"
                for i, q in enumerate(questions)
            ])
            
            prompt = f"""Review and improve these questions that lead to the answer: "{answer['text']}"

Context: {context}

Current Questions:
{questions_text}

For each question, suggest improvements considering:
1. Clarity and specificity
2. Natural language flow
3. Appropriateness for the answer
4. Question structure variety

Provide improved versions where needed in this format:
Original: [original question]
Improved: [improved version]
Reasoning: [why this is better]
Keep/Replace: [decision]

Begin review:"""

            # Get LLM response
            model_dict = self.llm
            if "vllm" in str(type(model_dict["model"])):
                response = self._get_vllm_response(model_dict["model"], prompt)
            else:
                response = model_dict["model"](prompt)

            # Parse refinements and update questions
            refined_questions = questions.copy()
            current_refinement = {}
            
            for line in response.split('\n'):
                line = line.strip()
                if not line:
                    if current_refinement.get('original') and current_refinement.get('improved'):
                        # Find and update matching question
                        for q in refined_questions:
                            if q['question'] == current_refinement['original']:
                                if current_refinement['decision'].lower() == 'replace':
                                    q['question'] = current_refinement['improved']
                                    q['refinement_reason'] = current_refinement['reasoning']
                                break
                        current_refinement = {}
                    continue
                
                if line.startswith('Original:'):
                    current_refinement['original'] = line[9:].strip()
                elif line.startswith('Improved:'):
                    current_refinement['improved'] = line[9:].strip()
                elif line.startswith('Reasoning:'):
                    current_refinement['reasoning'] = line[10:].strip()
                elif line.startswith('Keep/Replace:'):
                    current_refinement['decision'] = line[12:].strip()

            # Handle final refinement
            if current_refinement.get('original') and current_refinement.get('improved'):
                for q in refined_questions:
                    if q['question'] == current_refinement['original']:
                        if current_refinement['decision'].lower() == 'replace':
                            q['question'] = current_refinement['improved']
                            q['refinement_reason'] = current_refinement['reasoning']
                        break

            return refined_questions

        except Exception as e:
            logger.error(f"Error in question refinement: {e}")
            return questions

    def _validate_question(self, question: str, answer: str, context: str) -> bool:
        """Validate that a question is answerable from the context and matches the answer."""
        try:
            # Prepare validation prompt
            prompt = f"""Validate if this question-answer pair is valid based on the given context.

Context: {context}

Question: {question}
Proposed Answer: {answer}

Check the following:
1. Can the question be answered using ONLY the information in the context?
2. Is the proposed answer correct and complete for this question?
3. Is there a clear logical connection between the question and answer?

Respond with:
Valid: [true/false]
Reason: [explanation]

Analysis:"""

            # Get LLM response
            model_dict = self.llm
            if "vllm" in str(type(model_dict["model"])):
                response = self._get_vllm_response(model_dict["model"], prompt)
            else:
                response = model_dict["model"](prompt)

            # Parse validation result
            valid = False
            for line in response.split('\n'):
                if line.startswith('Valid:'):
                    valid = line[6:].strip().lower() == 'true'
                    break

            return valid

        except Exception as e:
            logger.error(f"Error in question validation: {e}")
            return False

    def _get_vllm_response(self, model: Any, prompt: str) -> str:
        """Get response from vLLM model."""
        outputs = model.generate([prompt], sampling_params=self.llm["sampling_params"])
        return outputs[0].outputs[0].text if outputs else ""

    def _generate_from_templates(
        self,
        answer: Dict[str, Any],
        content: str
    ) -> List[Dict[str, Any]]:
        """Generate questions from templates based on the answer and content."""
        qa_pairs = []
        
        # Get appropriate templates
        templates = self.templates["text"].get(
            answer["subtype"],
            self.templates["text"]["fact"]  # Default to fact templates
        )
        
        # Generate questions using templates
        for template in templates:
            # Extract subject for fact-based questions
            if answer["subtype"] == "fact":
                doc = self.nlp(answer["text"])
                subjects = [tok for tok in doc if tok.dep_ == "nsubj"]
                subject = subjects[0].text if subjects else "this"
                question = template.format(subject=subject)
            else:
                question = template.format(placeholder=answer["text"])
            
            qa_pairs.append({
                "question": question,
                "answer": answer["text"],
                "type": "text",
                "context": answer["context"],
                "score": answer["score"]
            })
        
        return qa_pairs
    
    def _generate_visual_questions(
        self,
        answers: List[Dict[str, Any]],
        content: str
    ) -> List[Dict[str, Any]]:
        """Generate questions for visual content answers."""
        qa_pairs = []
        
        for answer in answers:
            # Get appropriate templates
            templates = self.templates["visual"].get(
                answer["subtype"],
                self.templates["visual"]["caption"]  # Default to caption templates
            )
            
            # For visual QA answers, use the original question if available
            if answer["subtype"] == "analysis" and "question" in answer:
                qa_pairs.append({
                    "question": answer["question"],
                    "answer": answer["text"],
                    "type": "visual",
                    "context": answer["context"],
                    "score": answer["score"]
                })
                continue
            
            # Generate questions using templates
            for template in templates:
                qa_pairs.append({
                    "question": template,
                    "answer": answer["text"],
                    "type": "visual",
                    "context": answer["context"],
                    "score": answer["score"]
                })
        
        return qa_pairs
    
    def _generate_combined_questions(
        self,
        answers: List[Dict[str, Any]],
        content: str
    ) -> List[Dict[str, Any]]:
        """Generate questions that combine textual and visual information."""
        qa_pairs = []
        
        for answer in answers:
            # Extract the visual aspect from the combined answer
            visual_text = answer["text"]
            context_text = answer.get("context", "")
            
            # Get appropriate templates
            template_key = f"text_{answer['subtype']}"
            templates = self.templates["combined"].get(
                template_key,
                self.templates["combined"]["text_caption"]  # Default to caption templates
            )
            
            # Generate questions using templates
            for template in templates:
                # Find relevant context from the text content
                context = self._extract_relevant_context(context_text, visual_text)
                
                question = template.format(context=context)
                qa_pairs.append({
                    "question": question,
                    "answer": answer["text"],
                    "type": "combined",
                    "context": answer["context"],
                    "score": answer["score"]
                })
        
        return qa_pairs
    
    def _extract_relevant_context(self, context: str, answer: str) -> str:
        """Extract relevant context for combined questions."""
        # Split context into sentences
        doc = self.nlp(context)
        sentences = [sent.text for sent in doc.sents]
        
        if not sentences:
            return "this content"
        
        # Find most similar sentence to answer
        answer_embedding = self.sim_model.encode([answer])[0]
        sentence_embeddings = self.sim_model.encode(sentences)
        
        similarities = cosine_similarity(
            answer_embedding.reshape(1, -1),
            sentence_embeddings
        )[0]
        
        most_similar_idx = similarities.argmax()
        return sentences[most_similar_idx]
    
    def _filter_qa_pairs(self, qa_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter question-answer pairs based on quality criteria."""
        filtered = []
        
        for pair in qa_pairs:
            # Check question length
            q_words = len(pair["question"].split())
            if not (self.gen_config["min_question_length"] <= q_words <= self.gen_config["max_question_length"]):
                continue
            
            # Check answer length
            a_words = len(pair["answer"].split())
            if not (self.gen_config["min_answer_length"] <= a_words <= self.gen_config["max_answer_length"]):
                continue
            
            # Check minimum score
            if pair["score"] < self.gen_config["min_qa_score"]:
                continue
            
            # Remove duplicates
            if not self._is_duplicate_question(pair["question"], filtered):
                filtered.append(pair)
        
        return filtered
    
    def _rank_qa_pairs(self, qa_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank question-answer pairs by quality and diversity."""
        if not qa_pairs:
            return []
        
        # Calculate ranking scores
        for pair in qa_pairs:
            ranking_score = pair["score"]  # Start with base score
            
            # Adjust based on question type
            if self.gen_config.get("use_question_classifier"):
                q_type = self._classify_question(pair["question"])
                type_weights = {
                    "DESCRIPTION": 1.2,  # Prefer descriptive questions
                    "ENTITY": 1.1,      # Questions about specific entities
                    "HUMAN": 1.0,       # Questions about people
                    "LOCATION": 1.0,    # Questions about places
                    "NUMERIC": 0.9,     # Simple numeric questions
                    "ABBREVIATION": 0.8  # Simple abbreviation questions
                }
                ranking_score *= type_weights.get(q_type, 1.0)
            
            # Adjust based on type
            type_weights = {
                "combined": 1.2,  # Prefer questions that combine text and visual
                "visual": 1.1,    # Then visual questions
                "text": 1.0       # Then text questions
            }
            ranking_score *= type_weights.get(pair["type"], 1.0)
            
            pair["ranking_score"] = ranking_score
        
        # Sort by ranking score
        return sorted(qa_pairs, key=lambda x: x["ranking_score"], reverse=True)
    
    def _is_duplicate_question(self, question: str, existing: List[Dict[str, Any]]) -> bool:
        """Check if a question is semantically similar to existing questions."""
        if not existing:
            return False
            
        question_embedding = self.sim_model.encode([question])[0]
        
        for pair in existing:
            existing_embedding = self.sim_model.encode([pair["question"]])[0]
            similarity = cosine_similarity(
                question_embedding.reshape(1, -1),
                existing_embedding.reshape(1, -1)
            )[0][0]
            
            if similarity > self.gen_config["duplicate_threshold"]:
                return True
        
        return False
    
    def _classify_question(self, question: str) -> str:
        """Classify question type using pre-trained classifier."""
        try:
            inputs = self.q_tokenizer(
                question,
                return_tensors="pt",
                truncation=True,
                max_length=128
            )
            outputs = self.q_classifier(**inputs)
            predicted = outputs.logits.argmax(-1).item()
            return self.q_classifier.config.id2label[predicted]
        except Exception as e:
            logger.warning(f"Question classification failed: {e}")
            return "UNKNOWN"
    
    def _calculate_confidence(self, qa_pair: Dict[str, Any]) -> float:
        """Calculate overall confidence score for a QA pair."""
        confidence = qa_pair["score"]  # Start with answer confidence
        
        # Adjust based on question classification confidence
        if self.gen_config.get("use_question_classifier"):
            q_type = self._classify_question(qa_pair["question"])
            if q_type != "UNKNOWN":
                confidence *= 1.1  # Boost confidence for well-classified questions
        
        # Adjust based on context relevance
        if qa_pair["context"]:
            context_relevance = self._calculate_context_relevance(
                qa_pair["question"],
                qa_pair["answer"],
                qa_pair["context"]
            )
            confidence = (confidence + context_relevance) / 2
        
        return min(1.0, confidence)
    
    def _calculate_context_relevance(
        self,
        question: str,
        answer: str,
        context: str
    ) -> float:
        """Calculate relevance score between QA pair and its context."""
        qa_text = f"{question} {answer}"
        
        # Calculate semantic similarity
        qa_embedding = self.sim_model.encode([qa_text])[0]
        context_embedding = self.sim_model.encode([context])[0]
        
        similarity = cosine_similarity(
            qa_embedding.reshape(1, -1),
            context_embedding.reshape(1, -1)
        )[0][0]
        
        return float(similarity)