from typing import Dict, List, Any, Optional
import logging
import random
from langchain.prompts import PromptTemplate
from ..models.model_loader import ModelLoader

logger = logging.getLogger(__name__)

class SelfInstructGenerator:
    """Generates additional instruction data using self-instruction techniques."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the self-instruction generator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model_loader = ModelLoader(config)
        self.model = self.model_loader.get_model("self_instruct")
    
    def generate_instructions(
        self,
        content: str,
        seed_qa_pairs: List[Dict[str, Any]],
        num_pairs: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Generate new QA pairs using self-instruction.
        
        Args:
            content: Source content text
            seed_qa_pairs: List of seed QA pairs to guide generation
            num_pairs: Number of new pairs to generate
            
        Returns:
            List of generated QA pairs
        """
        try:
            # Format seed examples
            examples = self._format_examples(seed_qa_pairs)
            
            # Prepare self-instruction prompt
            prompt_template = PromptTemplate(
                template=(
                    "Given the following text, generate {num_pairs} new question-answer pairs. "
                    "The questions and answers must be derived ONLY from the information in the text. "
                    "Do not include any external knowledge.\n\n"
                    "Text:\n{content}\n\n"
                    "Here are some example question-answer pairs for reference:\n{examples}\n\n"
                    "Instructions:\n"
                    "1. Questions should be clear and specific\n"
                    "2. Answers must come directly from the text\n"
                    "3. Vary the types of questions (what, how, why, etc.)\n"
                    "4. Make questions progressively more complex\n"
                    "5. Format each pair as 'Q: [question] A: [answer]'\n\n"
                    "Generate {num_pairs} new question-answer pairs:"
                ),
                input_variables=["content", "examples", "num_pairs"]
            )
            
            # Generate new pairs
            prompt = prompt_template.format(
                content=content,
                examples=examples,
                num_pairs=num_pairs
            )
            
            # Get model response
            model = self.model["model"]
            response = model.predict(prompt) if hasattr(model, 'predict') else model(prompt)
            
            # Parse generated pairs
            generated_pairs = self._parse_qa_pairs(response, content)
            
            # Add metadata and validate
            validated_pairs = []
            for pair in generated_pairs:
                if self._validate_pair(pair, content):
                    pair["source"] = seed_qa_pairs[0]["source"]  # Use same source as seed
                    pair["generation_type"] = "self_instruct"
                    validated_pairs.append(pair)
            
            return validated_pairs[:num_pairs]  # Ensure we return requested number
            
        except Exception as e:
            logger.error(f"Error in self-instruction generation: {e}")
            return []
    
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