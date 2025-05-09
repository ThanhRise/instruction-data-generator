"""Helper functions for the instruction data generator."""

import logging
from typing import Dict, List, Any, Union, Optional, Callable
from pathlib import Path
import json
from datetime import datetime
import time

# Optional dependencies with fallbacks
try:
    import yaml
except ImportError:
    yaml = None

try:
    import torch
except ImportError:
    torch = None

try:
    import psutil
except ImportError:
    psutil = None

logger = logging.getLogger(__name__)

def setup_logging(log_dir: Union[str, Path], level: str = "INFO") -> None:
    """
    Set up logging configuration.
    
    Args:
        log_dir: Directory to store log files
        level: Logging level
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True, parents=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"instruction_generator_{timestamp}.log"
    
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def load_yaml_config(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML file with validation.
    
    Args:
        file_path: Path to YAML configuration file
        
    Returns:
        Dictionary containing configuration
    """
    if not yaml:
        raise ImportError("PyYAML is required for loading YAML files")
    
    try:
        with open(file_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Basic structure validation
        if not isinstance(config, dict):
            raise ValueError(f"Invalid config format in {file_path}. Expected dictionary.")
            
        return config
        
    except Exception as e:
        logger.error(f"Error loading YAML config from {file_path}: {e}")
        raise

def save_jsonl(data: List[Dict[str, Any]], file_path: Union[str, Path], append: bool = False) -> None:
    """
    Save data in JSONL format.
    
    Args:
        data: List of dictionaries to save
        file_path: Output file path
        append: Whether to append to existing file
    """
    try:
        mode = 'a' if append else 'w'
        with open(file_path, mode) as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
    except Exception as e:
        logger.error(f"Error saving JSONL to {file_path}: {e}")
        raise

def load_jsonl(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Load data from JSONL file.
    
    Args:
        file_path: Path to JSONL file
        
    Returns:
        List of dictionaries loaded from file
    """
    data = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
    except Exception as e:
        logger.error(f"Error loading JSONL from {file_path}: {e}")
        raise

def ensure_dir(dir_path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        dir_path: Directory path
        
    Returns:
        Path object for the directory
    """
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path

def get_file_type(file_path: Union[str, Path]) -> Optional[str]:
    """
    Get the type of a file based on its extension.
    
    Args:
        file_path: Path to file
        
    Returns:
        File type or None if unknown
    """
    file_path = Path(file_path)
    ext = file_path.suffix.lower()
    
    type_map = {
        '.txt': 'text',
        '.md': 'text',
        '.json': 'text',
        '.jsonl': 'text',
        '.csv': 'text',
        '.jpg': 'image',
        '.jpeg': 'image',
        '.png': 'image',
        '.bmp': 'image'
    }
    
    return type_map.get(ext)

def chunk_text(text: str, max_length: int = 512, overlap: int = 50) -> List[str]:
    """
    Split text into chunks with overlap.
    
    Args:
        text: Text to split
        max_length: Maximum chunk length
        overlap: Number of tokens to overlap
        
    Returns:
        List of text chunks
    """
    words = text.split()
    chunks = []
    
    if len(words) <= max_length:
        return [text]
    
    start = 0
    while start < len(words):
        end = start + max_length
        
        if end >= len(words):
            chunks.append(' '.join(words[start:]))
            break
            
        # Try to find sentence boundary
        for i in range(min(end + 20, len(words) - 1), max(end - 20, start), -1):
            if words[i].endswith(('.', '!', '?')):
                end = i + 1
                break
                
        chunks.append(' '.join(words[start:end]))
        start = end - overlap
        
    return chunks

def validate_config(config: Dict[str, Any], required_keys: List[str]) -> bool:
    """
    Validate configuration has required keys and correct structure.
    
    Args:
        config: Configuration dictionary
        required_keys: List of required top-level keys
        
    Returns:
        True if valid, False otherwise
    """
    # Check required keys
    missing = []
    for key in required_keys:
        if key not in config:
            missing.append(key)
            
    if missing:
        logger.error(f"Missing required configuration keys: {missing}")
        return False
    
    # Validate specific sections if they exist
    if "agent" in config:
        if not _validate_agent_config(config["agent"]):
            return False
            
    if "models" in config:
        if not _validate_model_config(config["models"]):
            return False
    
    return True

def _validate_agent_config(agent_config: Dict[str, Any]) -> bool:
    """
    Validate agent configuration section.
    
    Args:
        agent_config: Agent configuration dictionary
        
    Returns:
        True if valid, False otherwise
    """
    required_sections = [
        "input_processing",
        "instruction_generation",
        "quality_control",
        "document_processing"
    ]
    
    missing = []
    for section in required_sections:
        if section not in agent_config:
            missing.append(section)
            
    if missing:
        logger.error(f"Missing required agent config sections: {missing}")
        return False
        
    # Validate instruction generation settings
    instruction_gen = agent_config.get("instruction_generation", {})
    if not isinstance(instruction_gen.get("min_question_length", 0), int):
        logger.error("Invalid min_question_length in instruction_generation")
        return False
        
    if not isinstance(instruction_gen.get("max_question_length", 0), int):
        logger.error("Invalid max_question_length in instruction_generation")
        return False
        
    # Validate quality control settings
    quality_control = agent_config.get("quality_control", {})
    if not isinstance(quality_control.get("min_quality_score", 0), (int, float)):
        logger.error("Invalid min_quality_score in quality_control")
        return False
        
    return True

def _validate_model_config(model_config: Dict[str, Any]) -> bool:
    """
    Validate model configuration section.
    
    Args:
        model_config: Model configuration dictionary
        
    Returns:
        True if valid, False otherwise
    """
    # Check required sections
    required_sections = [
        "llm_models",
        "serving",
        "document_understanding",
        "image_processing"
    ]
    
    missing = []
    for section in required_sections:
        if section not in model_config:
            missing.append(section)
            
    if missing:
        logger.error(f"Missing required model config sections: {missing}")
        return False
        
    # Validate LLM models section
    llm_models = model_config.get("llm_models", {})
    for model_name, model_info in llm_models.items():
        if not isinstance(model_info, dict):
            logger.error(f"Invalid model info for {model_name}")
            return False
            
        # Check required model fields
        required_fields = ["name", "type"]
        missing_fields = [f for f in required_fields if f not in model_info]
        if missing_fields:
            logger.error(f"Model {model_name} missing required fields: {missing_fields}")
            return False
            
        # Validate model type
        if model_info["type"] not in ["api", "vllm", "transformers"]:
            logger.error(f"Invalid model type for {model_name}: {model_info['type']}")
            return False
            
    # Validate serving configuration
    serving = model_config.get("serving", {})
    if "vllm" in serving:
        vllm_config = serving["vllm"]
        required_vllm_fields = [
            "tensor_parallel_size",
            "gpu_memory_utilization",
            "dtype"
        ]
        missing_vllm = [f for f in required_vllm_fields if f not in vllm_config]
        if missing_vllm:
            logger.error(f"Missing required vLLM serving fields: {missing_vllm}")
            return False
            
    return True

class ModelMetricsLogger:
    def __init__(self, log_dir: Optional[str] = None):
        self.log_dir = Path(log_dir) if log_dir else Path("logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.performance_log = self.log_dir / "model_performance.json"
        self.gpu_log = self.log_dir / "gpu_metrics.json"
        self._initialize_logs()

    def _initialize_logs(self):
        """Initialize or load existing log files."""
        if not self.performance_log.exists():
            self._save_json(self.performance_log, {})
        if not self.gpu_log.exists():
            self._save_json(self.gpu_log, {})

    def _save_json(self, path: Path, data: Dict):
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def _load_json(self, path: Path) -> Dict:
        if path.exists():
            with open(path, 'r') as f:
                return json.load(f)
        return {}

    def log_model_usage(
        self,
        model_name: str,
        task_type: str,
        duration: float,
        input_tokens: int,
        output_tokens: int,
        success: bool,
        error: Optional[str] = None
    ):
        """Log detailed model usage metrics."""
        metrics = self._load_json(self.performance_log)
        
        if model_name not in metrics:
            metrics[model_name] = {"tasks": {}}
            
        if task_type not in metrics[model_name]["tasks"]:
            metrics[model_name]["tasks"][task_type] = []
            
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "duration": duration,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "success": success
        }
        
        if error:
            log_entry["error"] = error
            
        metrics[model_name]["tasks"][task_type].append(log_entry)
        self._save_json(self.performance_log, metrics)

    def log_gpu_stats(self, model_name: str):
        """Log GPU usage statistics if available."""
        try:
            import torch
            if not torch.cuda.is_available():
                return
                
            stats = self._load_json(self.gpu_log)
            if model_name not in stats:
                stats[model_name] = []
                
            current_stats = {
                "timestamp": datetime.now().isoformat(),
                "memory_allocated": torch.cuda.memory_allocated(),
                "memory_reserved": torch.cuda.memory_reserved(),
                "max_memory_allocated": torch.cuda.max_memory_allocated()
            }
            
            stats[model_name].append(current_stats)
            self._save_json(self.gpu_log, stats)
            
        except ImportError:
            logger.warning("PyTorch not available for GPU logging")
        except Exception as e:
            logger.error(f"Error logging GPU stats: {e}")

def timed_execution(func: Callable) -> Callable:
    """Decorator to time function execution."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            return result, duration
        except Exception as e:
            duration = time.time() - start_time
            raise e
    return wrapper

def estimate_tokens(text: str) -> int:
    """Roughly estimate the number of tokens in a text."""
    # Simple estimation: ~4 characters per token
    return len(text) // 4

def get_system_info() -> Dict[str, Any]:
    """Get system information for logging."""
    info = {}
    
    if psutil:
        info.update({
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_available": psutil.virtual_memory().available
        })
    
    if torch and torch.cuda.is_available():
        info["gpu"] = {
            "name": torch.cuda.get_device_name(),
            "memory_allocated": torch.cuda.memory_allocated(),
            "memory_reserved": torch.cuda.memory_reserved()
        }
    
    return info