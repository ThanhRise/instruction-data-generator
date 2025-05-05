try:
    import torch
    import transformers
    from vllm import LLM, SamplingParams
    from langchain.llms import OpenAI
    from langchain.chat_models import ChatOpenAI
    import bitsandbytes
    from typing import List
    HAS_CUDA = torch.cuda.is_available()
    NUM_GPUS = torch.cuda.device_count() if HAS_CUDA else 0
except ImportError as e:
    print(f"Warning: Some model dependencies not available: {e}")
    HAS_CUDA = False
    NUM_GPUS = 0

from typing import Dict, Any, Optional, Union
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class ConfigurationError(Exception):
    """Raised when there is an error in model configuration."""
    pass

class ModelLoadError(Exception):
    """Raised when there is an error loading a model."""
    pass

class ModelLoader:
    """Handles loading and configuration of different AI models with singleton pattern."""
    
    _instance = None
    _shared_llm = None  # Single shared LLM instance
    _initialized = False

    def __new__(cls, config: Dict[str, Any]):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
        return cls._instance

    def __init__(self, config: Dict[str, Any]):
        if not self._initialized:
            self.config = self._validate_config(config)
            self.vllm_instances = {}
            self.gpu_allocations = self._initialize_gpu_allocations()
            self._initialized = True

    def _initialize_gpu_allocations(self) -> Dict[str, List[int]]:
        """Initialize GPU allocations for different model types."""
        if not HAS_CUDA or NUM_GPUS == 0:
            return {}
            
        # Optimal GPU allocation strategy for 8x A100 40GB setup
        gpu_map = {
            # Large LLM models (70B+) - 6 GPUs
            "llama3_70b": list(range(6)),  # GPUs 0-5
            "llama2_70b": list(range(6)),  # GPUs 0-5
            "qwen25_72b": list(range(6)),  # GPUs 0-5
            "qwen2_70b": list(range(6)),  # GPUs 0-5
            
            # Smaller LLMs (13B/14B models) - 1 GPU
            "llama2_13b": [6],
            "qwen_14b": [6],
            "phi35": [6],
            
            # Vision models - 1 GPU
            "vision_models": [7]
        }
        return gpu_map

    def _get_gpu_device(self, model_name: str) -> Union[int, List[int]]:
        """Get assigned GPU device(s) for a model."""
        if not HAS_CUDA:
            return -1
            
        # Get base model name
        base_name = model_name.split('/')[-1].lower()
        
        # Map model names to their configurations
        model_patterns = {
            "llama-3": "llama3_70b",
            "llama-2-70b": "llama2_70b",
            "llama-2-13b": "llama2_13b",
            "qwen-2.5-72b": "qwen25_72b",
            "qwen-2-70b": "qwen2_70b",
            "qwen-14b": "qwen_14b",
            "phi-3.5": "phi35"
        }
        
        # Find matching pattern
        matched_model = None
        for pattern, model_key in model_patterns.items():
            if pattern in base_name:
                matched_model = model_key
                break
        
        if matched_model and matched_model in self.gpu_allocations:
            gpus = self.gpu_allocations[matched_model]
            return gpus[0] if len(gpus) == 1 else gpus
        
        # Default to last GPU for unknown models
        return self.gpu_allocations.get("vision_models", [-1])[0]

    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate model configuration."""
        if "models" not in config:
            raise ConfigurationError("No models section in configuration")
            
        models_config = config["models"]
        
        # Validate LLM models configuration
        if "llm_models" not in models_config:
            raise ConfigurationError("No llm_models section in configuration")
            
        for model_id, model_config in models_config["llm_models"].items():
            # Check required fields
            required_fields = ["name", "type"]
            missing = [f for f in required_fields if f not in model_config]
            if missing:
                raise ConfigurationError(
                    f"Model {model_id} missing required fields: {', '.join(missing)}"
                )
            
            # Validate model type
            valid_types = ["api", "vllm", "transformers"]
            if model_config["type"] not in valid_types:
                raise ConfigurationError(
                    f"Invalid model type for {model_id}: {model_config['type']}"
                )
            
            # Validate type-specific requirements
            if model_config["type"] == "vllm":
                if "model_path" not in model_config:
                    raise ConfigurationError(
                        f"vLLM model {model_id} requires model_path"
                    )
                
            # Validate parameters
            if "parameters" in model_config:
                params = model_config["parameters"]
                if not isinstance(params, dict):
                    raise ConfigurationError(
                        f"Parameters for {model_id} must be a dictionary"
                    )
        
        # Validate serving configuration if present
        if "serving" in models_config:
            serving_config = models_config["serving"]
            if "vllm" in serving_config:
                vllm_config = serving_config["vllm"]
                required_vllm_fields = [
                    "tensor_parallel_size",
                    "gpu_memory_utilization",
                    "dtype"
                ]
                missing = [f for f in required_vllm_fields if f not in vllm_config]
                if missing:
                    raise ConfigurationError(
                        f"vLLM serving configuration missing fields: {', '.join(missing)}"
                    )
        
        return config

    def get_model(self, model_type: str, model_name: Optional[str] = None, model_instance: Optional[Any] = None) -> Dict[str, Any]:
        """
        Get a model by type, loading it if not already cached.
        
        Args:
            model_type: Type of model to load (e.g., "question_generation")
            model_name: Optional specific model name to use instead of default
            model_instance: Optional pre-configured model instance to use instead of loading
            
        Returns:
            Dictionary containing model and its associated components
        """
        try:
            # If model instance is provided, use it directly
            if model_instance is not None:
                vllm_config = self.config["models"]["serving"]["vllm"]
                sampling_params = SamplingParams(
                    temperature=vllm_config.get("temperature", 0.8),
                    top_p=vllm_config.get("top_p", 0.95),
                    max_tokens=vllm_config.get("max_new_tokens", 150)
                )
                return {
                    "model": model_instance,
                    "sampling_params": sampling_params
                }
                
            cache_key = f"{model_type}_{model_name}" if model_name else model_type
            
            if cache_key in self._models:
                return self._models[cache_key]
                
            model_config = self.config["models"].get(model_type)
            if not model_config:
                raise ConfigurationError(f"No configuration found for model type: {model_type}")
            
            # Get specific model configuration
            if model_name:
                llm_config = self.config["models"]["llm_models"].get(model_name)
                if not llm_config:
                    raise ConfigurationError(f"No configuration found for model: {model_name}")
            else:
                default_model = model_config.get("default_model")
                if default_model:
                    llm_config = self.config["models"]["llm_models"].get(default_model)
                else:
                    llm_config = {"name": model_config["name"], "type": "transformers"}
            
            # Load model with proper error handling
            try:
                model = self._load_model(llm_config, model_config.get("parameters", {}))
                self._models[cache_key] = model
                return model
                
            except Exception as e:
                # Try fallback model if specified
                fallback = model_config.get("fallback_model")
                if fallback and fallback != model_name:
                    logger.warning(
                        f"Error loading model {llm_config['name']}, trying fallback: {fallback}"
                    )
                    return self.get_model(model_type, fallback)
                raise ModelLoadError(
                    f"Failed to load model {llm_config['name']}: {str(e)}"
                ) from e
                
        except Exception as e:
            raise ModelLoadError(f"Error getting model: {str(e)}") from e

    def _load_model(self, model_config: Dict[str, Any], task_params: Dict[str, Any]) -> Dict[str, Any]:
        """Load a model based on its configuration."""
        model_type = model_config["type"]
        model_name = model_config["name"]
        
        try:
            if model_type == "api":
                return self._load_api_model(model_name, {**model_config.get("parameters", {}), **task_params})
            elif model_type == "vllm":
                return self._load_vllm_model(model_config, task_params)
            elif model_type == "transformers":
                return self._load_transformers_model(model_name, {**model_config.get("parameters", {}), **task_params})
            else:
                raise ConfigurationError(f"Unsupported model type: {model_type}")
                
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load {model_type} model {model_name}: {str(e)}"
            ) from e

    def _validate_gpu_requirements(self, model_name: str) -> None:
        """Validate GPU requirements for a model."""
        try:
            if not HAS_CUDA:
                raise ModelLoadError(
                    f"Model {model_name} requires GPU but no CUDA device is available"
                )
            
            # Check memory requirements
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            required_memory = self._estimate_model_memory(model_name)
            
            if required_memory > gpu_memory:
                raise ModelLoadError(
                    f"Insufficient GPU memory for {model_name}. "
                    f"Required: {required_memory/1e9:.1f}GB, "
                    f"Available: {gpu_memory/1e9:.1f}GB"
                )
                
        except Exception as e:
            if not isinstance(e, ModelLoadError):
                raise ModelLoadError(f"Error validating GPU requirements: {str(e)}") from e
            raise

    def _estimate_model_memory(self, model_name: str) -> int:
        """Estimate memory requirements for a model."""
        # Model size estimates in bytes
        model_sizes = {
            "gpt-4": 0,  # API model
            "meta-llama/Llama-3.3-70b-chat-hf": 140e9,
            "meta-llama/Llama-2-70b-chat-hf": 140e9,
            "Qwen/Qwen-2.5-72B-Chat": 144e9,
            "Qwen/Qwen-2-70B-Chat": 140e9,
            "meta-llama/Llama-2-13b-chat-hf": 26e9,
            "Qwen/Qwen-14B-Chat": 28e9,
            "microsoft/phi-3.5": 7e9
        }
        
        # Get base model name
        base_name = model_name.split('/')[-1].lower()
        
        # Find matching model size
        for model_key, size in model_sizes.items():
            if model_key.lower().split('/')[-1] in base_name:
                return int(size * 2)  # 2x model size for safe margin
        
        # Default size for unknown models
        return int(10e9)

    def _try_load_in_8bit(self, model_name: str) -> bool:
        """Check if a model supports 8-bit quantization."""
        try:
            import bitsandbytes
            return True
        except ImportError:
            logger.warning(f"bitsandbytes not available, cannot load {model_name} in 8-bit")
            return False

    def get_shared_llm(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get the shared LLM instance, creating it if needed.
        This ensures a single LLM instance is shared across all components.
        
        Args:
            model_name: Optional specific model to use instead of default
            
        Returns:
            Dictionary containing model and its configuration
        """
        if self._shared_llm is None:
            logger.info(f"Creating shared LLM instance with model: {model_name}")
            
            # Get model configuration
            if model_name:
                if model_name not in self.config["models"]["llm_models"]:
                    raise ValueError(f"Model {model_name} not found in configuration")
                model_config = self.config["models"]["llm_models"][model_name]
            else:
                default_model = self.config["models"].get("default_model")
                if not default_model:
                    raise ValueError("No default model specified in configuration")
                model_config = self.config["models"]["llm_models"][default_model]
            
            # Initialize the model based on its type
            if model_config["type"] == "vllm":
                self._shared_llm = self._load_vllm_model(model_config, {})  # Pass empty dict as task_params
            elif model_config["type"] == "transformers":
                self._shared_llm = self._load_transformers_model(model_config["name"])
            elif model_config["type"] == "api":
                self._shared_llm = self._load_api_model(model_config["name"], model_config.get("parameters", {}))
            else:
                raise ValueError(f"Unsupported model type: {model_config['type']}")

        return self._shared_llm

    def clear_llm_cache(self) -> None:
        """Clear the shared LLM instance and GPU cache."""
        try:
            if self._shared_llm:
                # Clear model instance
                if "vllm" in str(type(self._shared_llm["model"])):
                    del self._shared_llm["model"]
                self._shared_llm = None
            
            # Clear vLLM instances
            for instance in self.vllm_instances.values():
                try:
                    del instance
                except Exception as e:
                    logger.warning(f"Error cleaning up vLLM instance: {e}")
            self.vllm_instances.clear()
            
            # Force CUDA cache clear if available
            if HAS_CUDA:
                torch.cuda.empty_cache()
                
        except Exception as e:
            logger.error(f"Error clearing LLM cache: {e}")

    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a model's configuration and status."""
        try:
            model_config = self.config["models"]["llm_models"].get(model_name)
            if not model_config:
                raise ConfigurationError(f"No configuration found for model: {model_name}")
            
            info = {
                "name": model_config["name"],
                "type": model_config["type"],
                "parameters": model_config.get("parameters", {}),
                "loaded": f"{model_name}" in str(self._models.keys()),
                "gpu_available": HAS_CUDA
            }
            
            if HAS_CUDA:
                info["gpu_memory"] = {
                    "total": torch.cuda.get_device_properties(0).total_memory,
                    "reserved": torch.cuda.memory_reserved(0),
                    "allocated": torch.cuda.memory_allocated(0)
                }
            
            return info
            
        except Exception as e:
            raise ModelLoadError(f"Error getting model info: {str(e)}") from e

    def _load_api_model(self, model_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Load API-based models (like GPT-4)."""
        if "chat" in model_name.lower():
            model = ChatOpenAI(
                model_name=model_name,
                temperature=params.get("temperature", 0.7),
                max_tokens=params.get("max_new_tokens", 150)
            )
        else:
            model = OpenAI(
                model_name=model_name,
                temperature=params.get("temperature", 0.7),
                max_tokens=params.get("max_new_tokens", 150)
            )
        return {"model": model}

    def _load_vllm_model(self, model_config: Dict[str, Any], task_params: Dict[str, Any]) -> Dict[str, Any]:
        """Load and configure model with vLLM."""
        model_path = model_config["model_path"]
        
        if model_path not in self.vllm_instances:
            # Get vLLM serving configuration
            vllm_config = self.config["models"]["serving"]["vllm"]
            
            # Get GPU allocation and tensor parallelism size
            gpu_devices = self._get_gpu_device(model_path)
            tensor_parallel_size = len(gpu_devices) if isinstance(gpu_devices, list) else 1
            
            # Setup model-specific tensor parallel size if specified
            model_tp_size = model_config.get("parameters", {}).get(
                "tensor_parallel_size",
                vllm_config["tensor_parallel_size"]
            )
            tensor_parallel_size = max(tensor_parallel_size, model_tp_size)
            
            # Initialize quantization settings if enabled
            quantization_config = None
            quantization = vllm_config.get("quantization", {})
            if quantization.get("enabled", False):
                quantization_config = {
                    "load_in_4bit": True,
                    "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_use_double_quant": quantization.get("use_double_quant", True),
                    "bnb_4bit_compute_dtype": torch.float16
                }
            
            # Initialize vLLM instance with optimized settings
            self.vllm_instances[model_path] = LLM(
                model=model_path,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=vllm_config["gpu_memory_utilization"],
                max_num_batched_tokens=vllm_config.get("max_num_batched_tokens", 8192),
                trust_remote_code=vllm_config.get("trust_remote_code", True),
                dtype=vllm_config["dtype"],
                quantization=quantization_config,
                max_model_len=8192,  # Increased context window
                enforce_eager=False,  # Better memory management
                seed=42,  # For reproducibility
            )
            
            logger.info(
                f"Initialized vLLM model {model_path} with tensor parallelism "
                f"size={tensor_parallel_size}, dtype={vllm_config['dtype']}"
            )
        
        # Combine model and task parameters
        params = {**model_config.get("parameters", {}), **task_params}
        
        # Create sampling parameters for inference
        sampling_params = SamplingParams(
            temperature=params.get("temperature", 0.8),
            top_p=params.get("top_p", 0.95),
            max_tokens=params.get("max_new_tokens", 150)
        )
        
        return {
            "model": self.vllm_instances[model_path],
            "sampling_params": sampling_params
        }

    def _load_transformers_model(self, model_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Load model using Hugging Face Transformers."""
        device = self._get_gpu_device(model_name)
        
        # Load tokenizer
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        
        # Load model with appropriate device mapping
        if isinstance(device, list):
            # For multi-GPU setups
            model = transformers.AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="balanced",
                max_memory={f"cuda:{i}": "35GB" for i in device}
            )
        else:
            # For single-GPU setups
            model = transformers.AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map={"": f"cuda:{device}" if device >= 0 else "cpu"}
            )
        
        return {"model": model, "tokenizer": tokenizer}