#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path
import os

from src.agent import InstructionDataGenerator
from huggingface_hub import login
import yaml
from dotenv import load_dotenv
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_huggingface_auth() -> None:
    """Setup Hugging Face authentication using token from environment variables."""
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        logger.warning("HF_TOKEN not found in environment variables. Some features may be limited.")
        return
    
    try:
        login(token=hf_token, write_permission=False)
        logger.info("Successfully logged in to Hugging Face Hub")
    except Exception as e:
        logger.error(f"Failed to login to Hugging Face Hub: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Run Instruction Data Generator with Multiple LLM Support")
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
        choices=[
            "llama2_70b",
            "llama3_70b",
            "qwen25_72b",
            "qwen2_70b",
            "llama2_13b",
            "qwen_14b",
            "phi35"
        ],
        help="Model name from config"
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
    parser.add_argument(
        "--max-memory",
        type=str,
        help="Max GPU memory per device (e.g., '35GiB')"
    )
    
    args = parser.parse_args()
    
    # Create necessary directories
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        # Load environment variables and setup HF auth
        load_dotenv()
        setup_huggingface_auth()
        
        # Set GPU memory limits if specified
        if args.max_memory:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = f"max_split_size_mb={(int(float(args.max_memory[:-3])*1024*0.8))}"
        
        # Initialize the agent
        agent = InstructionDataGenerator(
            config_path=args.config,
            model_name=args.model,
            log_dir=args.log_dir
        )
        
        # Start processing with enhanced monitoring
        logger.info(f"Starting instruction data generation with {args.model}...")
        try:
            agent.process_input_directory(
                input_dir=args.input_dir,
                output_dir=args.output_dir
            )
        except torch.cuda.OutOfMemoryError:
            logger.error("GPU Out of Memory error occurred. Consider using a model with lower memory requirements or adjusting batch size.")
            raise
        except Exception as e:
            logger.error(f"Error during processing: {str(e)}", exc_info=True)
            raise
        
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()