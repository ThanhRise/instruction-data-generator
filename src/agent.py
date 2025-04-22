from typing import Dict, List, Any, Optional, Union
import logging
import yaml
from pathlib import Path
import json

from .data_processing.data_loader import DataLoader
from .data_processing.image_annotator import ImageAnnotator
from .instruction_generation.answer_extractor import AnswerExtractor
from .instruction_generation.question_generator import QuestionGenerator
from .instruction_generation.self_instruct import SelfInstructGenerator
from .quality_control.quality_filter import QualityFilter

logger = logging.getLogger(__name__)

class InstructionDataGenerator:
    """Main agent class for generating instruction data from various input sources."""
    
    def __init__(self, config_path: str):
        """
        Initialize the instruction data generator agent.
        
        Args:
            config_path: Path to configuration file
        """
        config_dir = Path(config_path).parent
        self.config = self._load_config(
            config_path,
            config_dir / "model_config.yaml",
            env_file=config_dir / ".env"
        )
        
        # Initialize components
        self.data_loader = DataLoader(self.config)
        self.image_annotator = ImageAnnotator(self.config)
        self.answer_extractor = AnswerExtractor(self.config)
        self.question_generator = QuestionGenerator(self.config)
        self.self_instruct = SelfInstructGenerator(self.config)
        self.quality_filter = QualityFilter(self.config)
        
    def _load_config(self, *config_paths: Union[str, Path], env_file: Optional[Path] = None) -> Dict[str, Any]:
        """Load and merge configuration from YAML files and environment variables."""
        try:
            from utils.helpers import merge_configs
            return merge_configs(*config_paths, env_file=env_file)
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
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
            
            # Load and process input data
            input_data = self.data_loader.load_data(input_dir)
            
            # Process text and image data separately
            text_qa_pairs = self._process_text_data(input_data["text"])
            image_qa_pairs = self._process_image_data(input_data["images"])
            
            # Combine and filter all QA pairs
            all_qa_pairs = text_qa_pairs + image_qa_pairs
            
            # Save intermediate results if configured
            if self.config["agent"]["output"]["save_intermediate"]:
                self._save_intermediate_results(all_qa_pairs, output_path)
            
            # Save final instruction data
            self._save_instruction_data(all_qa_pairs, output_path)
            
            logger.info(f"Generated {len(all_qa_pairs)} instruction pairs")
            
        except Exception as e:
            logger.error(f"Error generating instruction data: {e}")
            raise
    
    def _process_text_data(self, text_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process text data to generate QA pairs."""
        qa_pairs = []
        contexts = {}
        
        try:
            # Extract answers from text
            for item in text_data:
                source = item["source"]
                content = item["content"]
                contexts[source] = content
                
                # Extract potential answers
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
            logger.error(f"Error processing text data: {e}")
            return []
    
    def _process_image_data(self, image_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process image data to generate QA pairs."""
        qa_pairs = []
        
        try:
            # Generate annotations for images
            annotated_data = self.image_annotator.annotate_images(
                image_data,
                Path(self.config["agent"]["output"]["save_intermediate"])
            )
            
            # Generate QA pairs from annotations
            for item in annotated_data:
                source = item["source"]
                annotation = item["annotation"]
                
                # Extract answers from annotation
                extracted_answers = self.answer_extractor.extract_answers(annotation)
                
                # Generate questions
                image_pairs = self.question_generator.generate_questions({
                    "source": source,
                    "content": annotation,
                    "extracted_answers": extracted_answers
                })
                
                qa_pairs.extend(image_pairs)
            
            # Filter pairs
            contexts = {item["source"]: item["annotation"] for item in annotated_data}
            filtered_pairs = self.quality_filter.filter_qa_pairs(qa_pairs, contexts)
            
            return filtered_pairs
            
        except Exception as e:
            logger.error(f"Error processing image data: {e}")
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