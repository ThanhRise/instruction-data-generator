from typing import Dict, List, Any
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
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize question generator."""
        self.config = config
        self.model_name = config["models"]["question_generation"]["name"]
        self.params = config["models"]["question_generation"]["parameters"]
        
        logger.info(f"Initializing question generator with model: {self.model_name}")
        
        if "t5" in self.model_name.lower():
            self.model = self._load_t5_model()
            self.model_type = "t5"
        elif "gpt" in self.model_name.lower() or "phi" in self.model_name.lower():
            self.model = self._load_llm_model()
            self.model_type = "llm"
        else:
            raise ValueError(f"Unsupported question generation model: {self.model_name}")
        
        # Load templates
        self.question_templates = self._load_question_templates()
    
    def _load_t5_model(self):
        """Load T5 model for question generation."""
        try:
            model_id = "flan-t5-xxl" if "flan-t5-xxl" in self.model_name else "t5-large"
            tokenizer = AutoTokenizer.from_pretrained(f"google/{model_id}")
            model = AutoModelForSeq2SeqLM.from_pretrained(
                f"google/{model_id}",
                torch_dtype=torch.float16,
                device_map="auto"
            )
            return {"model": model, "tokenizer": tokenizer}
        except Exception as e:
            logger.error(f"Error loading T5 model: {e}")
            raise
    
    def _load_llm_model(self):
        """Load LLM model for question generation."""
        try:
            if "gpt-4o" in self.model_name:
                from langchain_openai import ChatOpenAI
                model = ChatOpenAI(
                    model_name="gpt-4o",
                    temperature=self.params.get("temperature", 0.8)
                )
            elif "phi-3.5" in self.model_name:
                from langchain_huggingface import HuggingFaceEndpoint
                model = HuggingFaceEndpoint(
                    repo_id="microsoft/phi-3.5-instruct",
                    temperature=self.params.get("temperature", 0.8),
                    max_new_tokens=self.params.get("max_new_tokens", 150)
                )
            return {"model": model}
        except Exception as e:
            logger.error(f"Error loading LLM model: {e}")
            raise
    
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
        """
        Generate questions based on extracted answers and their contexts.
        
        Args:
            data: Dictionary containing answers, contexts, and document information
            
        Returns:
            List of generated QA pairs
        """
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
                
                # Generate questions based on content type
                if has_visual and "visual" in answer_type.lower():
                    questions = self._generate_visual_questions(
                        answer,
                        context,
                        doc_type,
                        answer_type
                    )
                else:
                    # Generate regular questions
                    if self.model_type == "t5":
                        questions = self._generate_with_t5(answer, context)
                    else:
                        questions = self._generate_with_llm(
                            answer,
                            context,
                            answer_type,
                            doc_type
                        )
                
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
                            "has_visual_context": has_visual
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
            
            # Generate additional questions using LLM
            if self.model_type == "llm":
                prompt = PromptTemplate(
                    template=(
                        "Generate a question about the visual content described in the following text. "
                        "The question should focus on what can be seen or understood from the visual element.\n\n"
                        "Visual description: {answer}\n"
                        "Document type: {doc_type}\n"
                        "Context: {context}\n\n"
                        "Generate a clear and specific question:"
                    ),
                    input_variables=["answer", "doc_type", "context"]
                )
                
                llm_response = self.model["model"](
                    prompt.format(
                        answer=answer,
                        doc_type=doc_type,
                        context=context
                    )
                )
                
                # Extract question from response
                if "?" in llm_response:
                    question = re.search(r'^.*?\?', llm_response)
                    if question:
                        questions.append(question.group(0))
            
            return questions[:3]  # Limit to 3 questions
            
        except Exception as e:
            logger.error(f"Error generating visual questions: {e}")
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
    
    def _generate_with_llm(
        self,
        answer: str,
        context: str,
        answer_type: str,
        doc_type: str = ""
    ) -> List[str]:
        """Generate questions using LLM model."""
        try:
            model = self.model["model"]
            
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
            if "gpt-4o" in self.model_name:
                response = model.predict(prompt)
            else:
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