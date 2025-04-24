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
    Load configuration from YAML file.
    
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

def load_env_config(env_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from environment file.
    
    Args:
        env_path: Path to environment file
        
    Returns:
        Dictionary containing environment configuration
    """
    try:
        from dotenv import load_dotenv
        import os
        
        # Load .env file
        load_dotenv(env_path)
        
        # Get environment variables
        env_config = {
            "api_keys": {
                "openai": os.getenv("OPENAI_API_KEY"),
                "huggingface": os.getenv("HF_TOKEN")
            },
            "endpoints": {
                "phi_model": os.getenv("PHI_MODEL_ENDPOINT")
            },
            "resources": {
                "max_gpu_memory": os.getenv("MAX_GPU_MEMORY", "12GB"),
                "max_cpu_memory": os.getenv("MAX_CPU_MEMORY", "32GB"),
                "num_workers": int(os.getenv("NUM_WORKERS", "4"))
            },
            "paths": {
                "cache_dir": os.getenv("CACHE_DIR", ".cache"),
                "model_dir": os.getenv("MODEL_DIR", "models"),
                "output_dir": os.getenv("OUTPUT_DIR", "data/output")
            }
        }
        
        return env_config
        
    except Exception as e:
        logger.error(f"Error loading environment config: {e}")
        raise

def merge_configs(*config_files: Union[str, Path], env_file: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Merge multiple configuration files with optional environment variables.
    Later configs override earlier ones.
    
    Args:
        *config_files: Paths to configuration files
        env_file: Optional path to environment file
        
    Returns:
        Merged configuration dictionary
    """
    if not yaml:
        raise ImportError("PyYAML is required for config file handling")
    
    merged = {}
    config_sources = {}  # Track which file each config key came from
    
    # Load and merge YAML configs
    for config_file in config_files:
        try:
            config = load_yaml_config(config_file)
            
            # Track conflicting keys
            for key in config:
                if key in merged:
                    # Check for potential conflicts
                    if isinstance(merged[key], dict) and isinstance(config[key], dict):
                        # Detect structural conflicts in nested dictionaries
                        conflicts = _detect_structural_conflicts(
                            merged[key], 
                            config[key], 
                            key,
                            config_sources[key],
                            str(config_file)
                        )
                        if conflicts:
                            raise ValueError(
                                f"Configuration conflict detected:\n" + "\n".join(conflicts)
                            )
                    config_sources[key] = f"{config_sources[key]}, {config_file}"
                else:
                    config_sources[key] = str(config_file)
            
            # Perform deep merge
            merged = _deep_merge_configs(merged, config)
            
        except Exception as e:
            logger.error(f"Error merging config from {config_file}: {e}")
            raise
    
    # Load and merge environment config if provided
    if env_file:
        try:
            env_config = load_env_config(env_file)
            
            # Update specific sections with environment values
            if "optimization" in merged:
                merged["optimization"]["max_memory"] = {
                    "gpu": env_config["resources"]["max_gpu_memory"],
                    "cpu": env_config["resources"]["max_cpu_memory"]
                }
            
            if "paths" in merged:
                merged["paths"].update(env_config["paths"])
                
            # Add API keys and endpoints section
            merged["api_keys"] = env_config["api_keys"]
            merged["endpoints"] = env_config["endpoints"]
            
        except Exception as e:
            logger.error(f"Error merging environment config: {e}")
            raise
            
    return merged

def _detect_structural_conflicts(
    dict1: Dict[str, Any],
    dict2: Dict[str, Any],
    path: str,
    source1: str,
    source2: str
) -> List[str]:
    """
    Detect structural conflicts between two dictionaries.
    
    Args:
        dict1: First dictionary
        dict2: Second dictionary
        path: Current path in the config hierarchy
        source1: Source file of first dictionary
        source2: Source file of second dictionary
        
    Returns:
        List of conflict messages, empty if no conflicts
    """
    conflicts = []
    
    for key in set(dict1.keys()) | set(dict2.keys()):
        current_path = f"{path}.{key}"
        
        # Check if key exists in both dictionaries
        if key in dict1 and key in dict2:
            val1, val2 = dict1[key], dict2[key]
            
            # Check for type mismatches
            if type(val1) != type(val2):
                conflicts.append(
                    f"Type mismatch at {current_path}:\n"
                    f"  {source1}: {type(val1).__name__}\n"
                    f"  {source2}: {type(val2).__name__}"
                )
            
            # Recursively check nested dictionaries
            elif isinstance(val1, dict) and isinstance(val2, dict):
                nested_conflicts = _detect_structural_conflicts(
                    val1, val2, current_path, source1, source2
                )
                conflicts.extend(nested_conflicts)
                
            # Check for value conflicts in non-dictionary types
            elif val1 != val2:
                conflicts.append(
                    f"Value conflict at {current_path}:\n"
                    f"  {source1}: {val1}\n"
                    f"  {source2}: {val2}"
                )
    
    return conflicts

def _deep_merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform a deep merge of two configuration dictionaries.
    
    Args:
        base: Base configuration dictionary
        override: Override configuration dictionary
        
    Returns:
        Merged configuration dictionary
    """
    merged = base.copy()
    
    for key, value in override.items():
        if (
            key in merged 
            and isinstance(merged[key], dict) 
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge_configs(merged[key], value)
        else:
            merged[key] = value
            
    return merged

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
        
    return True

def _validate_model_config(model_config: Dict[str, Any]) -> bool:
    """
    Validate model configuration section.
    
    Args:
        model_config: Model configuration dictionary
        
    Returns:
        True if valid, False otherwise
    """
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
        
    return True

class ModelMetricsLogger:
    """Tracks and logs model performance metrics."""
    
    def __init__(self, log_dir: Optional[str] = None):
        self.log_dir = Path(log_dir) if log_dir else Path("logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = {}
    
    def log_model_usage(
        self,
        model_name: str,
        task_type: str,
        duration: float,
        input_tokens: int,
        output_tokens: int,
        success: bool,
        error: Optional[str] = None
    ) -> None:
        """Log a model usage event."""
        if model_name not in self.metrics:
            self.metrics[model_name] = {
                "total_calls": 0,
                "successful_calls": 0,
                "failed_calls": 0,
                "total_duration": 0.0,
                "avg_duration": 0.0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "tasks": {},
                "errors": []
            }
        
        model_metrics = self.metrics[model_name]
        model_metrics["total_calls"] += 1
        model_metrics["successful_calls" if success else "failed_calls"] += 1
        model_metrics["total_duration"] += duration
        model_metrics["avg_duration"] = (
            model_metrics["total_duration"] / model_metrics["total_calls"]
        )
        model_metrics["total_input_tokens"] += input_tokens
        model_metrics["total_output_tokens"] += output_tokens
        
        if task_type not in model_metrics["tasks"]:
            model_metrics["tasks"][task_type] = {
                "calls": 0,
                "successes": 0,
                "failures": 0,
                "total_duration": 0.0
            }
        
        task_metrics = model_metrics["tasks"][task_type]
        task_metrics["calls"] += 1
        task_metrics["successes" if success else "failures"] += 1
        task_metrics["total_duration"] += duration
        
        if not success and error:
            model_metrics["errors"].append({
                "timestamp": time.time(),
                "task_type": task_type,
                "error": error
            })
        
        # Save metrics to file
        self._save_metrics()
    
    def log_gpu_stats(self, model_name: str) -> None:
        """Log GPU statistics for a model."""
        if not torch:
            return
            
        if not torch.cuda.is_available():
            return
        
        gpu_stats = {
            "memory_allocated": torch.cuda.memory_allocated(),
            "memory_reserved": torch.cuda.memory_reserved(),
            "max_memory_allocated": torch.cuda.max_memory_allocated()
        }
        
        if model_name not in self.metrics:
            self.metrics[model_name] = {}
        
        self.metrics[model_name]["gpu_stats"] = gpu_stats
        self._save_metrics()
    
    def get_model_performance(self, model_name: str) -> Dict[str, Any]:
        """Get performance metrics for a specific model."""
        return self.metrics.get(model_name, {})
    
    def _save_metrics(self) -> None:
        """Save metrics to a JSON file."""
        metrics_file = self.log_dir / "model_metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(self.metrics, f, indent=2)

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