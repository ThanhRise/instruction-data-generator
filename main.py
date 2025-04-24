#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path

from src.agent import InstructionDataGenerator
from vllm import LLM, SamplingParams
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model_config(config_path: str = "config/model_config.yaml") -> dict:
    """Load model configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_vllm_model(model_config: dict, model_name: str = "llama2_70b") -> LLM:
    """Initialize vLLM model with configuration."""
    model_settings = model_config["models"]["llm_models"].get(model_name)
    if not model_settings or model_settings["type"] != "vllm":
        raise ValueError(f"Model {model_name} not found or not a vLLM model")
    
    serving_config = model_config["models"]["serving"]["vllm"]
    
    llm = LLM(
        model=model_settings["name"],
        tensor_parallel_size=serving_config["tensor_parallel_size"],
        gpu_memory_utilization=serving_config["gpu_memory_utilization"],
        max_num_batched_tokens=serving_config["max_num_batched_tokens"],
        trust_remote_code=serving_config["trust_remote_code"],
        dtype=serving_config["dtype"],
        quantization_config={
            "load_in_4bit": serving_config["quantization"]["enabled"],
            "QuantizationConfig": {
                "bits": serving_config["quantization"]["bits"],
                "group_size": serving_config["quantization"]["group_size"]
            }
        } if serving_config["quantization"]["enabled"] else None,
    )
    return llm

def main():
    parser = argparse.ArgumentParser(description="Run Instruction Data Generator with Llama 3.3")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/agent_config.yaml",
        help="Path to agent configuration file"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="llama2_70b",
        help="Model name from config (default: llama2_70b)"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/input",
        help="Input directory containing source files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/output/instruction_data",
        help="Output directory for generated instruction data"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory for log files"
    )
    
    args = parser.parse_args()
    
    # Create necessary directories
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        # Load configurations
        model_config = load_model_config()
        
        # Initialize vLLM model
        llm = setup_vllm_model(model_config, args.model)
        logger.info(f"Initialized {args.model} with vLLM")
        
        # Initialize the agent with the model instance
        agent = InstructionDataGenerator(
            config_path=args.config,
            model_name=args.model,
            log_dir=args.log_dir,
            model_instance=llm  # Pass the vLLM model instance
        )
        
        # Start processing
        logger.info("Starting instruction data generation...")
        agent.process_input_directory(
            input_dir=args.input_dir,
            output_dir=args.output_dir
        )
        
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()