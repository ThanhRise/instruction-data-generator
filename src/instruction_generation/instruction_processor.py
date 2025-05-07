import time
from typing import Dict, List, Any, Optional, Tuple
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from ..models.model_loader import ModelLoader
from ..utils.helpers import chunk_text
import json
import re

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

    def generate_instruction_data(self, chunks: List[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Generate instruction data directly using LLM.
        
        Returns:
            Tuple containing:
            - List of successful instruction data pairs
            - List of failed pairs with failure reasons
        """
        instruction_data = []
        all_failed_pairs = []
        
        try:
            for chunk in chunks:
                # Generate QA pairs with instructions
                qa_pairs = self._generate_qa_pairs(chunk)
                instruction_data.extend(qa_pairs)
            
            if not instruction_data:
                logger.warning("No instruction data generated from chunks")
                return [], []
            
            # Filter duplicates
            unique_data = self._filter_duplicates(instruction_data)
            
            # Validate and get both successful and failed pairs
            validated_data, failed_pairs = self._validate_instruction_data(unique_data)
            
            # Log generation statistics
            logger.info(f"""Instruction data generation complete:
                Generated: {len(instruction_data)}
                After deduplication: {len(unique_data)}
                Validated: {len(validated_data)}
                Failed: {len(failed_pairs)}""")
            
            return validated_data, failed_pairs
            
        except Exception as e:
            logger.error(f"Error generating instruction data: {e}")
            return [], instruction_data  # Return all as failed if process errors out

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
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Validate generated instruction data with comprehensive checks.
        
        Returns:
            Tuple containing:
            - List of validated pairs that passed all checks
            - List of failed pairs with failure reasons
        """
        validated_pairs = []
        failed_pairs = []
        
        try:
            for pair in qa_pairs:
                # Skip invalid pairs
                if not all(pair.get(field) for field in ["question", "answer", "instruction", "context"]):
                    failed_pairs.append({
                        "pair": pair,
                        "reason": "Missing required fields",
                        "fields": [field for field in ["question", "answer", "instruction", "context"] if not pair.get(field)]
                    })
                    continue
                
                # Enhanced validation prompt with strict formatting
                validation_prompt = '''Carefully validate this instruction triplet against specific quality criteria.
Please analyze ONLY based on the provided context, without using external knowledge.

CONTEXT:
"""
{}
"""

TRIPLET TO VALIDATE:
Question: {}
Answer: {}
Instruction: {}

VALIDATION CRITERIA:
1. Answer Validity
- The answer MUST be fully supported by the context (no external information)
- The answer must be complete and accurately represent the information
- The answer should be relevant to the question

2. Question Quality
- The question must be clear, specific, and unambiguous
- The question should be answerable solely from the context
- The question should test understanding, not just fact recall
- The question must be grammatically correct

3. Instruction Quality
- Instructions must provide clear, step-by-step guidance
- Steps should lead logically to the correct answer
- Instructions must reference only information from the context
- Instructions should be actionable and clear

REQUIRED RESPONSE FORMAT:
Return ONLY a JSON object in this exact format, with no additional text:
{{
    "valid": boolean,
    "scores": {{
        "answer_validity": float,    // Score from 0-1
        "question_quality": float,   // Score from 0-1
        "instruction_quality": float // Score from 0-1
    }},
    "issues": [string],  // List of specific issues found, empty if valid
    "improvements": [string]  // Suggested improvements, empty if perfect
}}'''.format(
                    pair['context'],
                    pair['question'],
                    pair['answer'],
                    pair['instruction']
                )

                # Get and parse validation results with retries
                validation_result = self._get_validation_result(validation_prompt)
                
                if validation_result:
                    # Add validation scores to the pair
                    pair["validation_scores"] = validation_result.get("scores", {})
                    pair["validation_issues"] = validation_result.get("issues", [])
                    
                    # Check if pair meets quality thresholds
                    if (
                        validation_result.get("valid", False) and
                        self._meets_quality_thresholds(validation_result.get("scores", {}))
                    ):
                        validated_pairs.append(pair)
                    else:
                        failed_pairs.append({
                            "pair": pair,
                            "reason": "Failed quality validation",
                            "scores": validation_result.get("scores", {}),
                            "issues": validation_result.get("issues", []),
                            "improvements": validation_result.get("improvements", [])
                        })
                else:
                    failed_pairs.append({
                        "pair": pair,
                        "reason": "Validation parsing failed"
                    })
            
            # Log validation statistics
            total = len(qa_pairs)
            validated = len(validated_pairs)
            failed = len(failed_pairs)
            logger.info(f"Validation complete - Total: {total}, Passed: {validated}, Failed: {failed}")
            
            return validated_pairs, failed_pairs
            
        except Exception as e:
            logger.error(f"Error in validation: {e}")
            return [], qa_pairs  # Return all pairs as failed if validation errors out

    def _get_validation_result(self, prompt: str, max_retries: int = 3) -> Optional[Dict[str, Any]]:
        """Get and parse validation results with retry logic."""
        for attempt in range(max_retries):
            try:
                response = self._get_llm_response(prompt).strip()
                
                # Find JSON object in response using regex
                json_match = re.search(r'\{[\s\S]*\}', response)
                if json_match:
                    result = json.loads(json_match.group())
                    
                    # Validate response structure
                    if not self._is_valid_validation_response(result):
                        logger.warning(f"Invalid validation response structure on attempt {attempt + 1}")
                        continue
                    
                    return result
                    
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parsing failed on attempt {attempt + 1}: {e}")
            except Exception as e:
                logger.warning(f"Validation failed on attempt {attempt + 1}: {e}")
            
            # Wait briefly before retry
            time.sleep(1)
        
        return None

    def _is_valid_validation_response(self, response: Dict[str, Any]) -> bool:
        """Check if validation response has the required structure."""
        try:
            # Check required fields exist
            if not all(k in response for k in ["valid", "scores", "issues", "improvements"]):
                return False
            
            # Check scores structure
            required_scores = ["answer_validity", "question_quality", "instruction_quality"]
            if not all(k in response["scores"] for k in required_scores):
                return False
            
            # Validate score ranges
            for score in response["scores"].values():
                if not isinstance(score, (int, float)) or not 0 <= score <= 1:
                    return False
            
            # Validate other fields
            if not isinstance(response["valid"], bool):
                return False
            if not isinstance(response["issues"], list):
                return False
            if not isinstance(response["improvements"], list):
                return False
            
            return True
            
        except Exception:
            return False

    def _meets_quality_thresholds(self, scores: Dict[str, float]) -> bool:
        """Check if validation scores meet quality thresholds."""
        min_scores = {
            "answer_validity": self.proc_config.get("min_answer_validity_score", 0.7),
            "question_quality": self.proc_config.get("min_question_quality_score", 0.7),
            "instruction_quality": self.proc_config.get("min_instruction_quality_score", 0.7)
        }
        
        return all(
            scores.get(metric, 0) >= threshold
            for metric, threshold in min_scores.items()
        )

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