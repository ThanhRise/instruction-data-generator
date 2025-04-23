from typing import Dict, List, Any, Optional
import logging
import torch
import re
import random
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

logger = logging.getLogger(__name__)

class QuestionGenerator:
    """Generates questions based on extracted answers and their contexts."""
    
    def __init__(self, config: Dict[str, Any], model_name: Optional[str] = None, model_instance: Optional[Any] = None):
        """
        Initialize question generator.
        
        Args:
            config: Configuration dictionary
            model_name: Optional specific model to use instead of default
            model_instance: Optional pre-configured model instance to use instead of loading from config
        """
        self.config = config
        self.model_name = model_name if not model_instance else "custom_model"
        
        logger.info(f"Initializing question generator with model: {self.model_name}")
        
        # Initialize model loader
        from ..models.model_loader import ModelLoader
        self.model_loader = ModelLoader(config)
        
        # Load model components
        self.model = self.model_loader.get_model("question_generation", model_name, model_instance)
        
        # Load templates
        self.question_templates = self._load_question_templates()
    
    def _load_question_templates(self) -> Dict[str, List[str]]:
        """Load templates for different types of questions."""
        return {
            "visual_reference": [
                "What does {image_ref} show in the {doc_type}?",
                "Describe what can be seen in {image_ref}.",
                "What visual elements are present in {image_ref}?",
                "According to {image_ref}, what is being displayed?"
            ],
            "slide_content": [
                "What key points are presented about {topic}?",
                "What information does the slide provide about {topic}?",
                "What are the main ideas discussed in the slide about {topic}?",
                "How does the slide explain {topic}?"
            ],
            "document_fact": [
                "According to the {doc_type}, what {fact_type} is mentioned about {topic}?",
                "What specific information does the {doc_type} provide about {topic}?",
                "How does the {doc_type} describe {topic}?",
                "What details are given about {topic} in the {doc_type}?"
            ]
        }
    
    def generate_questions(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate questions based on extracted answers and their contexts."""
        qa_pairs = []
        
        try:
            source = data["source"]
            content = data.get("content", "")
            extracted_answers = data.get("extracted_answers", [])
            
            for answer_item in extracted_answers:
                answer = answer_item["answer"]
                answer_type = answer_item["type"]
                context = answer_item.get("context", "")
                
                # Get document-specific information
                doc_type = answer_item.get("document_type", "")
                has_visual = answer_item.get("has_visual_context", False)
                
                # Generate questions based on model type
                model_dict = self.model
                model_type = next(iter(model_dict["model"].__class__.__module__.split(".")))
                
                if has_visual and "visual" in answer_type.lower():
                    questions = self._generate_visual_questions(
                        answer,
                        context,
                        doc_type,
                        answer_type
                    )
                else:
                    if model_type == "vllm":
                        questions = self._generate_with_vllm(
                            model_dict["model"],
                            model_dict["sampling_params"],
                            answer,
                            context,
                            answer_type,
                            doc_type
                        )
                    elif model_type in ["langchain", "openai"]:
                        questions = self._generate_with_llm(
                            model_dict["model"],
                            answer,
                            context,
                            answer_type,
                            doc_type
                        )
                    elif model_type == "transformers":
                        questions = self._generate_with_transformers(
                            model_dict["model"],
                            model_dict["tokenizer"],
                            answer,
                            context
                        )
                    else:
                        questions = self._generate_with_t5(answer, context)
                
                # Validate and add QA pairs
                for question in questions:
                    if self._validate_question(question, answer, context):
                        qa_pairs.append({
                            "source": source,
                            "question": question,
                            "answer": answer,
                            "answer_type": answer_type,
                            "context": context,
                            "document_type": doc_type,
                            "has_visual_context": has_visual,
                            "model_used": self.model_name or "default"
                        })
            
            return qa_pairs
            
        except Exception as e:
            logger.error(f"Error generating questions: {e}")
            return []

    def _generate_visual_questions(
        self,
        answer: str,
        context: str,
        doc_type: str,
        answer_type: str
    ) -> List[str]:
        """Generate questions for visual content."""
        questions = []
        
        try:
            # Get appropriate templates
            templates = self.question_templates["visual_reference"]
            
            # Extract image reference terms
            image_refs = ["the figure", "the image", "the illustration", "the diagram"]
            if "chart" in answer.lower():
                image_refs.append("the chart")
            if "graph" in answer.lower():
                image_refs.append("the graph")
            
            # Generate questions using templates
            for template in templates:
                image_ref = random.choice(image_refs)
                question = template.format(
                    image_ref=image_ref,
                    doc_type=doc_type
                )
                questions.append(question)
            
            return questions[:3]  # Limit to 3 questions
            
        except Exception as e:
            logger.error(f"Error generating visual questions: {e}")
            return []

    def _generate_with_vllm(
        self,
        model: Any,
        sampling_params: Any,
        answer: str,
        context: str,
        answer_type: str,
        doc_type: str = ""
    ) -> List[str]:
        """Generate questions using vLLM model."""
        try:
            # Prepare prompt
            if doc_type:
                prompt = self._get_document_prompt(
                    context,
                    answer,
                    answer_type,
                    doc_type
                )
            else:
                prompt = self._get_standard_prompt(context, answer)
            
            # Generate completions
            outputs = model.generate([prompt], sampling_params)
            
            # Extract questions
            questions = []
            for output in outputs:
                generated_text = output.outputs[0].text
                questions.extend(self._extract_questions(generated_text))
            
            return questions[:3]  # Limit to 3 questions
            
        except Exception as e:
            logger.error(f"Error generating questions with vLLM: {e}")
            return []

    def _generate_with_transformers(
        self,
        model: Any,
        tokenizer: Any,
        answer: str,
        context: str
    ) -> List[str]:
        """Generate questions using Hugging Face transformers model."""
        try:
            prompt = self._get_standard_prompt(context, answer)
            
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True
            ).to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs["input_ids"],
                    max_new_tokens=150,
                    num_return_sequences=3,
                    temperature=0.8,
                    top_p=0.95,
                    do_sample=True
                )
            
            # Decode and process outputs
            questions = []
            for output in outputs:
                decoded = tokenizer.decode(output, skip_special_tokens=True)
                questions.extend(self._extract_questions(decoded))
            
            return questions
            
        except Exception as e:
            logger.error(f"Error generating questions with transformers: {e}")
            return []

    def _generate_with_llm(
        self,
        model: Any,
        answer: str,
        context: str,
        answer_type: str,
        doc_type: str = ""
    ) -> List[str]:
        """Generate questions using LLM model."""
        try:
            # Prepare prompt based on content type
            if doc_type:
                prompt_template = PromptTemplate(
                    template=(
                        "Generate three different questions based on the following content from "
                        "a {doc_type} document. The questions must be answerable using ONLY the "
                        "provided information.\n\n"
                        "Content: {context}\n"
                        "Answer to focus on: {answer}\n"
                        "Type of information: {answer_type}\n\n"
                        "Requirements:\n"
                        "1. Questions should be clear and specific\n"
                        "2. Questions should focus on the provided answer\n"
                        "3. Questions should match the document context\n"
                        "4. Each question should have a different focus or approach\n\n"
                        "Generate three questions:"
                    ),
                    input_variables=["doc_type", "context", "answer", "answer_type"]
                )
                
                prompt = prompt_template.format(
                    doc_type=doc_type,
                    context=context,
                    answer=answer,
                    answer_type=answer_type
                )
            else:
                # Use standard prompt for non-document content
                prompt_template = PromptTemplate(
                    template=(
                        "Generate three different questions that can be answered using ONLY the "
                        "following information. The questions must be specific and directly related "
                        "to the content.\n\n"
                        "Context: {context}\n"
                        "Answer to focus on: {answer}\n\n"
                        "Requirements:\n"
                        "1. Questions must be answerable solely from the given context\n"
                        "2. Questions should be clear and well-formed\n"
                        "3. Questions should vary in structure and complexity\n\n"
                        "Generate three questions:"
                    ),
                    input_variables=["context", "answer"]
                )
                
                prompt = prompt_template.format(
                    context=context,
                    answer=answer
                )
            
            # Generate questions
            response = model(prompt)
            
            # Extract questions from response
            questions = []
            lines = response.strip().split('\n')
            for line in lines:
                # Clean up numbered lists or bullet points
                line = re.sub(r'^[\d\-\.\)]+\s*', '', line.strip())
                if line and '?' in line:
                    # Extract the question part
                    question = re.search(r'^.*?\?', line)
                    if question:
                        questions.append(question.group(0).strip())
            
            # Post-process questions
            processed_questions = []
            for question in questions:
                # Ensure proper formatting
                if not question.strip().endswith('?'):
                    question = question.strip() + '?'
                question = question[0].upper() + question[1:]
                processed_questions.append(question)
            
            return processed_questions[:3]
            
        except Exception as e:
            logger.error(f"Error generating questions with LLM: {e}")
            return []

    def _generate_with_t5(self, answer: str, context: str) -> List[str]:
        """Generate questions using T5 model."""
        try:
            model = self.model["model"]
            tokenizer = self.model["tokenizer"]
            
            # Prepare input text
            input_text = f"answer: {answer} context: {context}"
            
            # Tokenize input
            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                max_length=512,
                truncation=True
            ).to(model.device)
            
            # Generate questions
            with torch.no_grad():
                outputs = model.generate(
                    inputs["input_ids"],
                    max_new_tokens=self.params.get("max_new_tokens", 150),
                    num_return_sequences=3,
                    temperature=self.params.get("temperature", 0.8),
                    top_p=self.params.get("top_p", 0.95),
                    do_sample=True
                )
            
            # Decode and clean up questions
            questions = []
            for output in outputs:
                question = tokenizer.decode(output, skip_special_tokens=True)
                
                # Remove prefixes and clean up
                question = re.sub(r'^(answer:|context:)\s*', '', question, flags=re.IGNORECASE)
                
                # Ensure proper formatting
                if not question.strip().endswith('?'):
                    question = question.strip() + '?'
                question = question[0].upper() + question[1:]
                
                questions.append(question)
            
            return questions
            
        except Exception as e:
            logger.error(f"Error generating questions with T5: {e}")
            return []

    def _get_standard_prompt(self, context: str, answer: str) -> str:
        """Get standard question generation prompt."""
        return self.config["models"]["prompts"]["question_generation"]["base_template"].format(
            context=context,
            answer=answer
        )

    def _get_document_prompt(
        self,
        context: str,
        answer: str,
        answer_type: str,
        doc_type: str
    ) -> str:
        """Get document-specific question generation prompt."""
        return self.config["models"]["prompts"]["document_question"]["base_template"].format(
            doc_type=doc_type,
            content=context,
            focus=answer,
            section=answer_type
        )

    def _extract_questions(self, text: str) -> List[str]:
        """Extract questions from generated text."""
        questions = []
        lines = text.strip().split('\n')
        
        for line in lines:
            # Clean up numbered lists or bullet points
            line = re.sub(r'^[\d\-\.\)]+\s*', '', line.strip())
            if line and '?' in line:
                # Extract the question part
                question = re.search(r'^.*?\?', line)
                if question:
                    questions.append(question.group(0).strip())
        
        # Post-process questions
        processed = []
        for question in questions:
            # Ensure proper formatting
            if not question.strip().endswith('?'):
                question = question.strip() + '?'
            question = question[0].upper() + question[1:]
            processed.append(question)
        
        return processed

    def _validate_question(self, question: str, answer: str, context: str) -> bool:
        """Validate generated question."""
        # Basic validation
        if not question or len(question.split()) < 3:
            return False
        
        if not question.strip().endswith('?'):
            return False
        
        # Check for question-answer validity
        question_lower = question.lower()
        answer_lower = answer.lower()
        
        # Avoid questions that contain the answer
        if answer_lower in question_lower:
            return False
        
        # Check answer presence in context
        if answer_lower not in context.lower():
            return False
        
        # Check for common question words
        question_words = {"what", "who", "where", "when", "why", "how", "which", "whose"}
        if not any(word in question_lower.split() for word in question_words):
            return False
        
        return True