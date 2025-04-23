from typing import Dict, List, Any, Optional
import logging
import random
import torch
from langchain.prompts import PromptTemplate
from ..models.model_loader import ModelLoader

logger = logging.getLogger(__name__)

class SelfInstructGenerator:
    """Generates additional instruction data using self-instruction techniques."""
    
    def __init__(self, config: Dict[str, Any], model_name: Optional[str] = None, model_instance: Optional[Any] = None):
        """
        Initialize the self-instruction generator.
        
        Args:
            config: Configuration dictionary
            model_name: Optional specific model to use instead of default
            model_instance: Optional pre-configured model instance to use instead of loading from config
        """
        self.config = config
        self.model_name = model_name if not model_instance else "custom_model"
        
        if model_instance:
            # Use provided model instance
            self.model = {
                "model": model_instance,
                "sampling_params": self.config["models"]["serving"]["vllm"]
            }
        else:
            # Initialize model loader and load from config
            self.model_loader = ModelLoader(config)
            self.model = self.model_loader.get_model("self_instruct", model_name)
    
    def generate_instructions(
        self,
        content: str,
        seed_qa_pairs: List[Dict[str, Any]],
        num_pairs: int = 5
    ) -> List[Dict[str, Any]]:
        """Generate new QA pairs using self-instruction."""
        try:
            # Format seed examples
            examples = self._format_examples(seed_qa_pairs)
            
            # Get model components
            model_dict = self.model
            model_type = next(iter(model_dict["model"].__class__.__module__.split(".")))
            
            # Generate based on model type
            if model_type == "vllm":
                generated_pairs = self._generate_with_vllm(
                    model_dict["model"],
                    model_dict["sampling_params"],
                    content,
                    examples,
                    num_pairs
                )
            elif model_type in ["langchain", "openai"]:
                generated_pairs = self._generate_with_llm(
                    model_dict["model"],
                    content,
                    examples,
                    num_pairs
                )
            elif model_type == "transformers":
                generated_pairs = self._generate_with_transformers(
                    model_dict["model"],
                    model_dict["tokenizer"],
                    content,
                    examples,
                    num_pairs
                )
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Add metadata and validate
            validated_pairs = []
            for pair in generated_pairs:
                if self._validate_pair(pair, content):
                    pair["source"] = seed_qa_pairs[0]["source"]  # Use same source as seed
                    pair["generation_type"] = "self_instruct"
                    pair["model_used"] = self.model_name or "default"
                    validated_pairs.append(pair)
            
            return validated_pairs[:num_pairs]
            
        except Exception as e:
            logger.error(f"Error in self-instruction generation: {e}")
            return []

    def _generate_with_vllm(
        self,
        model: Any,
        sampling_params: Any,
        content: str,
        examples: str,
        num_pairs: int
    ) -> List[Dict[str, Any]]:
        """Generate instruction pairs using vLLM."""
        try:
            # Prepare prompt
            prompt = self._get_generation_prompt(content, examples, num_pairs)
            
            # Generate completions
            outputs = model.generate([prompt], sampling_params)
            
            # Parse generated pairs
            pairs = []
            for output in outputs:
                generated_text = output.outputs[0].text
                pairs.extend(self._parse_qa_pairs(generated_text, content))
            
            return pairs
            
        except Exception as e:
            logger.error(f"Error generating with vLLM: {e}")
            return []

    def _generate_with_transformers(
        self,
        model: Any,
        tokenizer: Any,
        content: str,
        examples: str,
        num_pairs: int
    ) -> List[Dict[str, Any]]:
        """Generate instruction pairs using Hugging Face transformers."""
        try:
            prompt = self._get_generation_prompt(content, examples, num_pairs)
            
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                max_length=1024,
                truncation=True
            ).to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs["input_ids"],
                    max_new_tokens=512,
                    num_return_sequences=2,
                    temperature=0.9,
                    top_p=0.95,
                    do_sample=True
                )
            
            # Decode and parse outputs
            pairs = []
            for output in outputs:
                decoded = tokenizer.decode(output, skip_special_tokens=True)
                pairs.extend(self._parse_qa_pairs(decoded, content))
            
            return pairs
            
        except Exception as e:
            logger.error(f"Error generating with transformers: {e}")
            return []

    def _generate_with_llm(
        self,
        model: Any,
        content: str,
        examples: str,
        num_pairs: int
    ) -> List[Dict[str, Any]]:
        """Generate instruction pairs using LLM."""
        try:
            prompt = self._get_generation_prompt(content, examples, num_pairs)
            
            # Generate pairs
            response = model(prompt)
            
            # Parse generated pairs
            return self._parse_qa_pairs(response, content)
            
        except Exception as e:
            logger.error(f"Error generating with LLM: {e}")
            return []

    def _get_generation_prompt(
        self,
        content: str,
        examples: str,
        num_pairs: int
    ) -> str:
        """Get prompt for instruction generation."""
        return self.config["models"]["prompts"]["self_instruct"]["base_template"].format(
            text=content,
            examples=examples,
            num_pairs=num_pairs
        )

    def _format_examples(self, qa_pairs: List[Dict[str, Any]]) -> str:
        """Format QA pairs as examples for the prompt."""
        examples = []
        for pair in qa_pairs:
            examples.append(f"Q: {pair['question']}\nA: {pair['answer']}")
        return "\n\n".join(examples)
    
    def _parse_qa_pairs(self, text: str, context: str) -> List[Dict[str, Any]]:
        """Parse generated text into QA pairs."""
        pairs = []
        current_pair = {}
        
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('Q:'):
                # Save previous pair if exists
                if current_pair.get('question') and current_pair.get('answer'):
                    pairs.append(current_pair.copy())
                # Start new pair
                current_pair = {
                    'question': line[2:].strip(),
                    'context': context
                }
            elif line.startswith('A:') and current_pair.get('question'):
                current_pair['answer'] = line[2:].strip()
        
        # Add last pair if complete
        if current_pair.get('question') and current_pair.get('answer'):
            pairs.append(current_pair)
        
        return pairs
    
    def _validate_pair(self, pair: Dict[str, Any], context: str) -> bool:
        """Validate a generated QA pair."""
        question = pair.get('question', '').strip()
        answer = pair.get('answer', '').strip()
        
        if not question or not answer:
            return False
            
        # Check question format
        if not question.endswith('?'):
            return False
            
        # Check answer presence in context
        answer_lower = answer.lower()
        context_lower = context.lower()
        
        # Allow for some variation in answer phrasing
        words = answer_lower.split()
        if len(words) <= 3:
            # For short answers, require exact match
            if answer_lower not in context_lower:
                return False
        else:
            # For longer answers, check if most words appear in context
            word_presence = [word in context_lower for word in words]
            if sum(word_presence) / len(word_presence) < 0.8:
                return False
        
        return True
    
    def augment_with_variations(
        self, 
        qa_pairs: List[Dict[str, Any]], 
        num_variations: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Generate variations of existing QA pairs.
        
        Args:
            qa_pairs: Original QA pairs
            num_variations: Number of variations to generate per pair
            
        Returns:
            List including original and variant pairs
        """
        try:
            augmented_pairs = []
            for pair in qa_pairs:
                augmented_pairs.append(pair)  # Keep original
                
                # Generate variations
                variations = self._generate_variations(
                    pair["question"],
                    pair["answer"],
                    pair["context"],
                    num_variations
                )
                
                # Add valid variations
                for var in variations:
                    if self._validate_pair(var, pair["context"]):
                        var["source"] = pair["source"]
                        var["generation_type"] = "variation"
                        augmented_pairs.append(var)
            
            return augmented_pairs
            
        except Exception as e:
            logger.error(f"Error generating variations: {e}")
            return qa_pairs
    
    def _generate_variations(
        self,
        question: str,
        answer: str,
        context: str,
        num_variations: int
    ) -> List[Dict[str, Any]]:
        """Generate variations of a QA pair."""
        try:
            prompt_template = PromptTemplate(
                template=(
                    "Generate {num_variations} different versions of the following question-answer pair. "
                    "Keep the same meaning but vary the phrasing and complexity.\n\n"
                    "Context: {context}\n"
                    "Original Question: {question}\n"
                    "Original Answer: {answer}\n\n"
                    "Requirements:\n"
                    "1. Questions must remain answerable from the context\n"
                    "2. Answers must maintain the same factual content\n"
                    "3. Vary question structures and vocabulary\n"
                    "Format each variation as 'Q: [question] A: [answer]'\n\n"
                    "Generate variations:"
                ),
                input_variables=["context", "question", "answer", "num_variations"]
            )
            
            prompt = prompt_template.format(
                context=context,
                question=question,
                answer=answer,
                num_variations=num_variations
            )
            
            # Generate variations
            model = self.model["model"]
            response = model.predict(prompt) if hasattr(model, 'predict') else model(prompt)
            
            # Parse variations
            variations = self._parse_qa_pairs(response, context)
            
            return variations
            
        except Exception as e:
            logger.error(f"Error generating QA variations: {e}")
            return []