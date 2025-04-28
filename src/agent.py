from typing import Dict, List, Any, Optional, Union
import logging
from pathlib import Path
import json

from .models.model_loader import ModelLoader

# Optional dependencies with fallbacks
try:
    import yaml
except ImportError:
    yaml = None

from .data_processing.data_loader import DataLoader
from .data_processing.image_annotator import ImageAnnotator
from .instruction_generation.answer_extractor import AnswerExtractor
from .instruction_generation.question_generator import QuestionGenerator
from .instruction_generation.self_instruct import SelfInstructGenerator
from .quality_control.quality_filter import QualityFilter
from .utils.helpers import (
    _validate_agent_config,
    _validate_model_config,
    load_env_config,
    load_yaml_config,
    setup_logging,
    ModelMetricsLogger,
    timed_execution,
    estimate_tokens,
    merge_configs,
    validate_config
)

logger = logging.getLogger(__name__)

class InstructionDataGenerator:
    """Main agent class for generating instruction data from various input sources."""
    
    def __init__(
        self,
        agent_config_path: str,
        model_config_path: str,
        model_name: Optional[str] = None,
        log_dir: Optional[str] = None
    ):
        """
        Initialize the instruction data generator agent.
        
        Args:
            agent_config_path: Path to agent configuration file
            model_config_path: Path to model configuration file
            model_name: Optional name of the LLM model to use
            log_dir: Optional directory for logging
        """
        if not yaml:
            raise ImportError("PyYAML is required for configuration handling")
            
        # Load and merge configurations
        self.config = self._load_and_validate_configs(
            agent_config_path,
            model_config_path
        )
        
        # Set up logging
        setup_logging(log_dir or "logs")
        self.metrics_logger = ModelMetricsLogger(log_dir)
        
        # Initialize ModelLoader first to ensure singleton pattern
        self.model_loader = ModelLoader(self.config)
        
        # Initialize components with shared model
        self.data_loader = DataLoader(self.config)
        self.image_annotator = ImageAnnotator(self.config)
        self.answer_extractor = AnswerExtractor(self.config, model_name)
        self.question_generator = QuestionGenerator(self.config, model_name)
        self.self_instruct = SelfInstructGenerator(self.config, model_name)
        self.quality_filter = QualityFilter(self.config)
        
        # Track current model
        self.current_model = model_name

    def _load_and_validate_configs(
        self,
        agent_config_path: str,
        model_config_path: str
    ) -> Dict[str, Any]:
        """
        Load and validate agent and model configurations.
        
        Args:
            agent_config_path: Path to agent configuration
            model_config_path: Path to model configuration
            
        Returns:
            Merged configuration dictionary
        """
        try:
            # Load individual configs first
            agent_config = load_yaml_config(agent_config_path)
            model_config = load_yaml_config(model_config_path)
            
            # Validate individual configs
            if not _validate_agent_config(agent_config.get("agent", {})):
                raise ValueError("Invalid agent configuration")
                
            if not _validate_model_config(model_config.get("models", {})):
                raise ValueError("Invalid model configuration")
            
            # Merge configurations
            merged_config = {
                "agent": agent_config.get("agent", {}),
                "models": model_config.get("models", {}),
                "paths": agent_config.get("paths", {})
            }
            
            return merged_config
            
        except Exception as e:
            logger.error(f"Error loading configurations: {e}")
            raise

    def switch_model(self, model_name: str) -> None:
        """Switch to a different LLM model."""
        if model_name not in self.config["models"]["llm_models"]:
            raise ValueError(f"Model {model_name} not found in configuration")
            
        try:
            # Clear existing model cache
            self.model_loader.clear_llm_cache()
            
            # Reinitialize components with new model
            self.answer_extractor = AnswerExtractor(self.config, model_name)
            self.question_generator = QuestionGenerator(self.config, model_name)
            self.self_instruct = SelfInstructGenerator(self.config, model_name)
            
            self.current_model = model_name
            logger.info(f"Switched to model: {model_name}")
            
        except Exception as e:
            logger.error(f"Error switching model: {e}")
            raise
    
    def get_model_performance(self) -> Dict[str, Any]:
        """Get performance metrics for current model."""
        if not self.current_model:
            return {}
        return self.metrics_logger.get_model_performance(self.current_model)
    
    @timed_execution
    def generate_instruction_data(self, input_dir: str, output_dir: str) -> None:
        """
        Generate instruction data from input directory.
        
        Args:
            input_dir: Directory containing input data
            output_dir: Directory to save generated instruction data
        """
        try:
            # Create output directory
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Load input data - now returns unified documents
            input_data = self.data_loader.load_data(input_dir)
            
            # Process all documents with unified content
            qa_pairs = self._process_unified_content(input_data["documents"])
            
            # Log model performance
            if self.current_model:
                self.metrics_logger.log_model_usage(
                    model_name=self.current_model,
                    task_type="instruction_generation",
                    duration=0.0,  # Will be updated by decorator
                    input_tokens=sum(estimate_tokens(doc["content"]) for doc in input_data["documents"]),
                    output_tokens=sum(estimate_tokens(str(pair)) for pair in qa_pairs),
                    success=True
                )
                # Log GPU stats if available
                self.metrics_logger.log_gpu_stats(self.current_model)
            
            # Save intermediate results if configured
            if self.config["agent"]["output"]["save_intermediate"]:
                self._save_intermediate_results(qa_pairs, output_path)
            
            # Save final instruction data
            self._save_instruction_data(qa_pairs, output_path)
            
            logger.info(f"Generated {len(qa_pairs)} instruction pairs")
            
        except Exception as e:
            if self.current_model:
                self.metrics_logger.log_model_usage(
                    model_name=self.current_model,
                    task_type="instruction_generation",
                    duration=0.0,
                    input_tokens=0,
                    output_tokens=0,
                    success=False,
                    error=str(e)
                )
            logger.error(f"Error generating instruction data: {e}")
            raise

    def _load_config(self, *config_paths: Union[str, Path], env_file: Optional[Path] = None) -> Dict[str, Any]:
        """Load and merge configuration from YAML files and environment variables."""
        try:
            return merge_configs(*config_paths, env_file=env_file)
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    def _process_unified_content(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process unified document content to generate QA pairs."""
        qa_pairs = []
        contexts = {}
        
        try:
            # Extract answers from unified content
            for doc in documents:
                source = doc["source"]
                content = doc["content"]
                contexts[source] = content
                
                # Extract potential answers from unified content
                extracted_answers = self.answer_extractor.extract_answers(content)
                
                # Generate questions for extracted answers
                initial_pairs = self.question_generator.generate_questions({
                    "source": source,
                    "content": content,
                    "extracted_answers": extracted_answers
                })
                
                qa_pairs.extend(initial_pairs)
            
            # Generate additional pairs using self-instruction
            augmented_pairs = []
            for source, content in contexts.items():
                source_pairs = [p for p in qa_pairs if p["source"] == source]
                if source_pairs:
                    additional_pairs = self.self_instruct.generate_instructions(
                        content,
                        source_pairs[:3],  # Use top 3 pairs as seeds
                        num_pairs=5
                    )
                    augmented_pairs.extend(additional_pairs)
            
            qa_pairs.extend(augmented_pairs)
            
            # Filter and validate pairs
            filtered_pairs = self.quality_filter.filter_qa_pairs(qa_pairs, contexts)
            validated_pairs = self.quality_filter.validate_against_source(filtered_pairs, contexts)
            
            return validated_pairs
            
        except Exception as e:
            logger.error(f"Error processing unified content: {e}")
            return []
    
    def _save_intermediate_results(self, qa_pairs: List[Dict[str, Any]], output_dir: Path) -> None:
        """Save intermediate processing results."""
        try:
            intermediate_dir = output_dir / "intermediate"
            intermediate_dir.mkdir(exist_ok=True)
            
            # Group QA pairs by source
            by_source = {}
            for pair in qa_pairs:
                source = pair["source"]
                if source not in by_source:
                    by_source[source] = []
                by_source[source].append(pair)
            
            # Save intermediate files
            for source, pairs in by_source.items():
                source_path = Path(source)
                output_file = intermediate_dir / f"{source_path.stem}_qa_pairs.json"
                
                with open(output_file, 'w') as f:
                    json.dump(pairs, f, indent=2)
                    
        except Exception as e:
            logger.error(f"Error saving intermediate results: {e}")
    
    def _save_instruction_data(self, qa_pairs: List[Dict[str, Any]], output_dir: Path) -> None:
        """Save final instruction data."""
        try:
            output_format = self.config["agent"]["output"]["format"]
            output_file = output_dir / f"instruction_data.{output_format}"
            
            if output_format == "jsonl":
                with open(output_file, 'w') as f:
                    for pair in qa_pairs:
                        f.write(json.dumps(pair) + '\n')
            else:
                with open(output_file, 'w') as f:
                    json.dump(qa_pairs, f, indent=2)
                    
            logger.info(f"Saved instruction data to {output_file}")
            
        except Exception as e:
            logger.error(f"Error saving instruction data: {e}")
            raise
    
    def get_statistics(self, qa_pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about generated instruction data."""
        try:
            stats = {
                "total_pairs": len(qa_pairs),
                "by_source": {},
                "avg_question_length": 0,
                "avg_answer_length": 0,
                "question_types": {}
            }
            
            # Compute statistics
            question_lengths = []
            answer_lengths = []
            
            for pair in qa_pairs:
                source = pair["source"]
                if source not in stats["by_source"]:
                    stats["by_source"][source] = 0
                stats["by_source"][source] += 1
                
                question_lengths.append(len(pair["question"].split()))
                answer_lengths.append(len(pair["answer"].split()))
                
                # Count question types
                first_word = pair["question"].strip().lower().split()[0]
                if first_word not in stats["question_types"]:
                    stats["question_types"][first_word] = 0
                stats["question_types"][first_word] += 1
            
            # Calculate averages
            if question_lengths:
                stats["avg_question_length"] = sum(question_lengths) / len(question_lengths)
            if answer_lengths:
                stats["avg_answer_length"] = sum(answer_lengths) / len(answer_lengths)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error computing statistics: {e}")
            return {}