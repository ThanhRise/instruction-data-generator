from typing import Dict, List, Any, Optional
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from ..models.model_loader import ModelLoader
from ..utils.helpers import chunk_text
import json

logger = logging.getLogger(__name__)

class InstructionDataProcessor:
    """Processes input data and generates instruction data using LLM."""
    
    def __init__(self, config: Dict[str, Any], model_name: Optional[str] = None):
        self.config = config
        self.proc_config = config["agent"]["instruction_generation"]
        
        # Initialize components
        self._initialize_components()
        
        # Get shared LLM instance
        self.model_loader = ModelLoader(config)
        self.model_name = model_name if model_name else self.config["models"]["llm_models"].get("default", "gpt-4")
        self.llm = self.model_loader.get_shared_llm(self.model_name)

    def _initialize_components(self):
        """Initialize NLP components."""
        try:
            # Initialize semantic search model for content relevance
            self.sim_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            
            # Initialize text splitter for semantic chunking
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.proc_config.get("chunk_size", 1500),
                chunk_overlap=self.proc_config.get("chunk_overlap", 150),
                separators=["\n\n", "\n", ". ", ", ", " "],
                length_function=len
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise

    def process_input_data(self, content: str) -> List[Dict[str, Any]]:
        """Process input data to generate high-quality chunks."""
        try:
            # Initial chunking
            raw_chunks = self.text_splitter.split_text(content)
            
            # Score and filter chunks for relevance
            scored_chunks = self._score_chunks_relevance(raw_chunks)
            
            # Filter out low-quality chunks
            quality_threshold = self.proc_config.get("chunk_quality_threshold", 0.6)
            relevant_chunks = [
                chunk for chunk, score in scored_chunks 
                if score > quality_threshold
            ]
            
            # Merge short related chunks if needed
            processed_chunks = self._merge_related_chunks(relevant_chunks)
            
            return processed_chunks
            
        except Exception as e:
            logger.error(f"Error processing input data: {e}")
            return []

    def _score_chunks_relevance(self, chunks: List[str]) -> List[tuple[str, float]]:
        """Score chunks based on information density and coherence."""
        scored_chunks = []
        
        try:
            # Get embeddings for all chunks
            embeddings = self.sim_model.encode(chunks)
            
            for i, chunk in enumerate(chunks):
                # Calculate information density score
                density_score = self._calculate_info_density(chunk)
                
                # Calculate coherence with other chunks
                coherence_score = self._calculate_chunk_coherence(
                    embeddings[i],
                    embeddings,
                    i
                )
                
                # Combine scores
                final_score = 0.6 * density_score + 0.4 * coherence_score
                scored_chunks.append((chunk, final_score))
            
            return scored_chunks
            
        except Exception as e:
            logger.error(f"Error scoring chunks: {e}")
            return [(chunk, 0.0) for chunk in chunks]

    def _calculate_info_density(self, text: str) -> float:
        """Calculate information density score for a chunk of text."""
        try:
            # Use LLM to evaluate information content
            prompt = f"""Rate the information density and usefulness of this text for instruction data generation.
Consider:
1. Presence of factual content
2. Clear concepts or relationships
3. Absence of boilerplate or filler text
4. Potential for generating meaningful Q&A pairs

Text: {text}

Respond with only a score between 0 and 1:"""

            response = self._get_llm_response(prompt)
            try:
                score = float(response.strip())
                return min(max(score, 0.0), 1.0)
            except:
                return 0.5
                
        except Exception as e:
            logger.error(f"Error calculating info density: {e}")
            return 0.5

    def _calculate_chunk_coherence(
        self,
        chunk_embedding: np.ndarray,
        all_embeddings: np.ndarray,
        current_idx: int
    ) -> float:
        """Calculate semantic coherence score for a chunk."""
        try:
            # Get similarity with other chunks
            similarities = cosine_similarity(
                chunk_embedding.reshape(1, -1),
                all_embeddings
            )[0]
            
            # Remove self-similarity
            similarities = np.delete(similarities, current_idx)
            
            # Take average of top 3 similarities
            top_k = min(3, len(similarities))
            if top_k > 0:
                coherence = np.mean(np.sort(similarities)[-top_k:])
            else:
                coherence = 0.0
                
            return float(coherence)
            
        except Exception as e:
            logger.error(f"Error calculating coherence: {e}")
            return 0.0

    def _merge_related_chunks(self, chunks: List[str]) -> List[str]:
        """Merge short, semantically related chunks."""
        if not chunks:
            return []
            
        try:
            # Get embeddings
            embeddings = self.sim_model.encode(chunks)
            
            # Calculate pairwise similarities
            similarities = cosine_similarity(embeddings)
            
            # Find chunks to merge
            merged_chunks = []
            skip_indices = set()
            
            for i in range(len(chunks)):
                if i in skip_indices:
                    continue
                    
                current_chunk = chunks[i]
                
                # Find highly similar neighbors
                neighbors = []
                for j in range(len(chunks)):
                    if i != j and j not in skip_indices:
                        if similarities[i][j] > 0.8:  # High similarity threshold
                            neighbors.append(j)
                
                if neighbors:
                    # Merge with similar chunks
                    merged = current_chunk
                    for j in neighbors:
                        merged += "\n" + chunks[j]
                        skip_indices.add(j)
                    merged_chunks.append(merged)
                else:
                    merged_chunks.append(current_chunk)
            
            return merged_chunks
            
        except Exception as e:
            logger.error(f"Error merging chunks: {e}")
            return chunks

    def generate_instruction_data(self, chunks: List[str]) -> List[Dict[str, Any]]:
        """Generate instruction data directly using LLM."""
        instruction_data = []
        
        try:
            for chunk in chunks:
                # Generate QA pairs with instructions
                qa_pairs = self._generate_qa_pairs(chunk)
                instruction_data.extend(qa_pairs)
            
            # Filter duplicates and validate
            unique_data = self._filter_duplicates(instruction_data)
            validated_data = self._validate_instruction_data(unique_data)
            
            return validated_data
            
        except Exception as e:
            logger.error(f"Error generating instruction data: {e}")
            return []

    def _generate_qa_pairs(self, text: str) -> List[Dict[str, Any]]:
        """Generate question-answer pairs with instructions using LLM."""
        try:
            # System prompt to set context and constraints
            system_prompt = """You are an expert instruction data generator.
Your task is to create high-quality question-answer-instruction triplets from given text.
Each triplet must be:
1. Directly derived from the input text (no external knowledge)
2. Clear and specific
3. Structured to promote understanding and reasoning
4. Properly formatted according to the specified structure

Follow these guidelines:
- Questions should require understanding, not just fact lookup
- Answers must be fully supported by the text
- Instructions should explain the reasoning process step-by-step
- Generate diverse question types (factual, inferential, analytical)
- Ensure all information comes from the provided text"""

            # Generation prompt with clear structure and examples
            generation_prompt = f"""Given this text, generate 3-5 high-quality instruction triplets.

Text to analyze:
{text}

For each key piece of information, create a triplet with:
1. Question (Q): A clear, specific question that tests understanding
2. Answer (A): The correct answer, fully supported by the text
3. Instruction (I): Step-by-step guidance on how to find or derive the answer

Use this exact format for each triplet:

---START_TRIPLET---
QUESTION: [Your question here]
ANSWER: [Your answer here]
INSTRUCTION: [Your step-by-step instruction here]
---END_TRIPLET---

Example structure (do not use this content, generate from the provided text):
---START_TRIPLET---
QUESTION: What was the key innovation that led to the company's success in 1995?
ANSWER: The development of their patented compression algorithm
INSTRUCTION: 1. Look for mentions of technological innovations
2. Identify the specific development in 1995
3. Connect this development to the company's success
---END_TRIPLET---

Remember:
- Use clear section markers
- Include all three components
- Make instructions detailed and step-by-step
- Ensure everything comes from the text

Generate the triplets now:"""

            # Get LLM response with the enhanced prompts
            response = self._get_llm_response(f"{system_prompt}\n\n{generation_prompt}")
            return self._parse_qa_pairs(response, text)
            
        except Exception as e:
            logger.error(f"Error generating QA pairs: {e}")
            return []

    def _parse_qa_pairs(self, response: str, context: str) -> List[Dict[str, Any]]:
        """Parse LLM response into structured QA pairs with robust error handling."""
        pairs = []
        try:
            # Split into individual triplets
            triplets = response.split("---START_TRIPLET---")
            for triplet in triplets:
                if "---END_TRIPLET---" not in triplet:
                    continue
                    
                triplet = triplet.split("---END_TRIPLET---")[0].strip()
                if not triplet:
                    continue
                
                # Initialize triplet dict with required fields
                current_pair = {
                    "question": None,
                    "answer": None,
                    "instruction": None,
                    "context": context
                }
                
                # Parse each field with error handling
                for line in triplet.split('\n'):
                    line = line.strip()
                    if not line:
                        continue
                        
                    try:
                        if line.startswith("QUESTION:"):
                            current_pair["question"] = line[9:].strip()
                        elif line.startswith("ANSWER:"):
                            current_pair["answer"] = line[7:].strip()
                        elif line.startswith("INSTRUCTION:"):
                            # Handle multi-line instructions
                            instruction_lines = []
                            instruction_lines.append(line[12:].strip())
                            
                            # Look ahead for numbered steps
                            for next_line in triplet.split('\n')[triplet.split('\n').index(line) + 1:]:
                                if next_line.strip() and (
                                    next_line.strip()[0].isdigit() or 
                                    not any(next_line.startswith(f) for f in ["QUESTION:", "ANSWER:", "INSTRUCTION:"])
                                ):
                                    instruction_lines.append(next_line.strip())
                                else:
                                    break
                            
                            current_pair["instruction"] = "\n".join(instruction_lines)
                    
                    except Exception as e:
                        logger.warning(f"Error parsing line '{line}': {e}")
                        continue
                
                # Validate all required fields are present and non-empty
                if all(current_pair[field] for field in ["question", "answer", "instruction"]):
                    pairs.append(current_pair)
                else:
                    logger.warning(f"Skipping incomplete triplet: {current_pair}")
            
            return pairs
            
        except Exception as e:
            logger.error(f"Error parsing QA pairs: {e}")
            return []

    def _filter_duplicates(self, qa_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate or highly similar QA pairs."""
        if not qa_pairs:
            return []
            
        unique_pairs = []
        question_embeddings = self.sim_model.encode(
            [pair['question'] for pair in qa_pairs]
        )
        
        for i, pair in enumerate(qa_pairs):
            # Check similarity with existing unique pairs
            is_duplicate = False
            for j in range(i):
                if j >= len(unique_pairs):
                    break
                similarity = cosine_similarity(
                    question_embeddings[i].reshape(1, -1),
                    question_embeddings[j].reshape(1, -1)
                )[0][0]
                if similarity > 0.85:  # High similarity threshold
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_pairs.append(pair)
        
        return unique_pairs

    def _validate_instruction_data(
        self,
        qa_pairs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Validate generated instruction data with comprehensive checks."""
        validated_pairs = []
        
        try:
            for pair in qa_pairs:
                # Skip invalid pairs
                if not all(pair.get(field) for field in ["question", "answer", "instruction", "context"]):
                    logger.warning(f"Skipping validation of incomplete pair: {pair}")
                    continue
                
                # Comprehensive validation prompt
                validation_prompt = f"""Validate this instruction triplet for quality and completeness:

Context: {pair['context']}

Question: {pair['question']}
Answer: {pair['answer']}
Instruction: {pair['instruction']}

Validation criteria:
1. Answer Validity:
   - Is the answer fully supported by the context?
   - Is the answer complete and accurate?

2. Question Quality:
   - Is the question clear and unambiguous?
   - Is it specific to the context?
   - Does it require understanding (not just fact lookup)?

3. Instruction Effectiveness:
   - Are the steps clear and logical?
   - Do they lead to the correct answer?
   - Are they grounded in the context?

4. Overall Coherence:
   - Do the question, answer, and instruction work together?
   - Is everything derived purely from the context?

Respond with a JSON object:
{
    "valid": true/false,
    "scores": {
        "answer_validity": 0-1,
        "question_quality": 0-1,
        "instruction_quality": 0-1
    }
}"""

                # Get validation results
                response = self._get_llm_response(validation_prompt)
                try:
                    validation_result = json.loads(response)
                    
                    # Check overall validity and quality thresholds
                    if (
                        validation_result.get("valid", False) and
                        all(
                            validation_result.get("scores", {}).get(metric, 0) >= 0.7
                            for metric in ["answer_validity", "question_quality", "instruction_quality"]
                        )
                    ):
                        validated_pairs.append(pair)
                    else:
                        logger.info(f"Pair failed validation: {validation_result}")
                        
                except json.JSONDecodeError:
                    logger.warning(f"Invalid validation response format: {response}")
                    continue
                
            return validated_pairs
            
        except Exception as e:
            logger.error(f"Error in validation: {e}")
            return []

    def _get_llm_response(self, prompt: str) -> str:
        """Get response from LLM."""
        try:
            model_dict = self.llm
            if "vllm" in str(type(model_dict["model"])):
                outputs = model_dict["model"].generate(
                    [prompt],
                    self.llm["sampling_params"]
                )
                return outputs[0].outputs[0].text if outputs else ""
            else:
                return model_dict["model"](prompt)
        except Exception as e:
            logger.error(f"Error getting LLM response: {e}")
            return ""