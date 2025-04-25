#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path
import os

from src.agent import InstructionDataGenerator
from src.utils.helpers import load_env_config
from vllm import LLM, SamplingParams
from huggingface_hub import login
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_huggingface_auth() -> None:
    """Setup Hugging Face authentication using token from environment config."""
    env_config = load_env_config()
    hf_token = env_config.get("api_keys", {}).get("huggingface")
    
    if not hf_token:
        logger.warning("Hugging Face token not found in environment config. Some features may be limited.")
        return
    
    try:
        login(token=hf_token, write_permission=False)
        logger.info("Successfully logged in to Hugging Face Hub")
    except Exception as e:
        logger.error(f"Failed to login to Hugging Face Hub: {e}")
        raise

def load_model_config() -> dict:
    """Load model configuration from config file."""
    try:
        with open("config/model_config.yaml", "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading model config: {e}")
        raise

def setup_vllm_model(model_config: dict, model_name: str) -> LLM:
    """Setup vLLM model with configuration."""
    model_params = model_config.get("models", {}).get(model_name)
    if not model_params:
        raise ValueError(f"Model {model_name} not found in config")
        
    try:
        return LLM(
            model=model_params["path"],
            trust_remote_code=True,
            tensor_parallel_size=model_params.get("tensor_parallel_size", 1),
            dtype=model_params.get("dtype", "auto"),
            gpu_memory_utilization=model_params.get("gpu_memory_utilization", 0.9)
        )
    except Exception as e:
        logger.error(f"Error initializing vLLM model: {e}")
        raise

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
        # Load environment config and setup HF auth
        setup_huggingface_auth()
        
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