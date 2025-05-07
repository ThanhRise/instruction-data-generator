from typing import Dict, List, Any, Optional
import logging
from pathlib import Path
import json

from .models.model_loader import ModelLoader
from .data_processing.data_loader import DataLoader
from .data_processing.image_annotator import ImageAnnotator
from .instruction_generation.instruction_processor import InstructionDataProcessor
from .quality_control.quality_filter import QualityFilter
from .utils.helpers import (
    _validate_agent_config,
    _validate_model_config,
    load_yaml_config,
    setup_logging,
    ModelMetricsLogger,
    timed_execution,
    estimate_tokens
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
        # Load and validate configurations
        self.config = self._load_and_validate_configs(
            agent_config_path,
            model_config_path
        )
        
        # Set up logging
        setup_logging(log_dir or "logs")
        self.metrics_logger = ModelMetricsLogger(log_dir)
        
        # Initialize components
        self.model_loader = ModelLoader(self.config)
        self.data_loader = DataLoader(self.config)
        self.image_annotator = ImageAnnotator(self.config)
        self.instruction_processor = InstructionDataProcessor(self.config, model_name)
        self.quality_filter = QualityFilter(self.config)
        
        self.current_model = model_name

    def _load_and_validate_configs(
        self,
        agent_config_path: str,
        model_config_path: str
    ) -> Dict[str, Any]:
        """Load and validate configurations."""
        try:
            agent_config = load_yaml_config(agent_config_path)
            model_config = load_yaml_config(model_config_path)
            
            if not _validate_agent_config(agent_config.get("agent", {})):
                raise ValueError("Invalid agent configuration")
                
            if not _validate_model_config(model_config.get("models", {})):
                raise ValueError("Invalid model configuration")
            
            return {
                "agent": agent_config.get("agent", {}),
                "models": model_config.get("models", {}),
                "paths": agent_config.get("paths", {})
            }
            
        except Exception as e:
            logger.error(f"Error loading configurations: {e}")
            raise

    @timed_execution
    def generate_instruction_data(self, input_dir: str, output_dir: str) -> None:
        """Generate instruction data from input directory."""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Load and unify input data
            input_data = self.data_loader.load_data(input_dir)
            
            # Process each document
            all_instruction_data = []
            all_failed_pairs = []
            
            for doc in input_data["documents"]:
                # Process input data to get high-quality chunks
                processed_chunks = self.instruction_processor.process_input_data(doc["content"])
                
                # Generate instruction data from chunks
                doc_instructions, failed_pairs = self.instruction_processor.generate_instruction_data(processed_chunks)
                
                # Add source information
                for item in doc_instructions:
                    item["source"] = doc["source"]
                for item in failed_pairs:
                    if isinstance(item, dict) and "pair" in item:
                        item["pair"]["source"] = doc["source"]
                    else:
                        # Handle case where the entire item is the failed pair
                        item["source"] = doc["source"]
                
                all_instruction_data.extend(doc_instructions)
                all_failed_pairs.extend(failed_pairs)
            
            # Final quality filtering
            final_data = self.quality_filter.filter_qa_pairs(
                all_instruction_data,
                {doc["source"]: doc["content"] for doc in input_data["documents"]}
            )
            
            # Log metrics
            if self.current_model:
                self.metrics_logger.log_model_usage(
                    model_name=self.current_model,
                    task_type="instruction_generation",
                    duration=0.0,  # Will be updated by decorator
                    input_tokens=sum(estimate_tokens(doc["content"]) for doc in input_data["documents"]),
                    output_tokens=sum(estimate_tokens(str(pair)) for pair in final_data),
                    success=True
                )
                self.metrics_logger.log_gpu_stats(self.current_model)
            
            # Save successful instruction data
            self._save_instruction_data(final_data, output_path)
            
            # Save failed pairs for analysis if needed
            if all_failed_pairs:
                failed_path = output_path / "failed_pairs"
                failed_path.mkdir(exist_ok=True)
                self._save_instruction_data(all_failed_pairs, failed_path)
            
            # Log generation statistics
            logger.info(f"""Instruction data generation complete:
                Total generated: {len(all_instruction_data) + len(all_failed_pairs)}
                Successfully validated: {len(all_instruction_data)}
                Failed validation: {len(all_failed_pairs)}
                After quality filtering: {len(final_data)}""")
            
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
    
    def _save_instruction_data(self, data: List[Dict[str, Any]], output_dir: Path) -> None:
        """Save generated instruction data."""
        try:
            output_format = self.config["agent"]["output"]["format"]
            output_file = output_dir / f"instruction_data.{output_format}"
            
            if output_format == "jsonl":
                with open(output_file, 'w') as f:
                    for item in data:
                        f.write(json.dumps(item) + '\n')
            else:
                with open(output_file, 'w') as f:
                    json.dump(data, f, indent=2)
                    
            logger.info(f"Saved instruction data to {output_file}")
            
        except Exception as e:
            logger.error(f"Error saving instruction data: {e}")
            raise