from typing import Dict, Any, Optional
import logging
from pathlib import Path
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
    AutoModelForSeq2SeqLM
)
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

logger = logging.getLogger(__name__)

class ModelLoader:
    """Handles loading and configuration of different AI models."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the model loader.
        
        Args:
            config: Configuration dictionary containing model settings
        """
        self.config = config
        self.models_cache = {}
        
    def get_model(self, model_type: str) -> Dict[str, Any]:
        """
        Get a model by type, loading it if not already cached.
        
        Args:
            model_type: Type of model to load (e.g., "question_generation", "image_annotation")
            
        Returns:
            Dictionary containing model and its associated components
        """
        if model_type in self.models_cache:
            return self.models_cache[model_type]
            
        model_config = self.config["models"].get(model_type)
        if not model_config:
            raise ValueError(f"No configuration found for model type: {model_type}")
            
        model_name = model_config["name"]
        params = model_config.get("parameters", {})
        
        try:
            if "gpt-4" in model_name.lower():
                model = self._load_openai_model(model_name, params)
            elif "phi-3.5" in model_name.lower():
                model = self._load_phi_model(model_name, params)
            elif "t5" in model_name.lower():
                model = self._load_t5_model(model_name, params)
            elif "blip" in model_name.lower():
                model = self._load_blip_model(model_name, params)
            else:
                raise ValueError(f"Unsupported model: {model_name}")
                
            self.models_cache[model_type] = model
            return model
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            raise
    
    def _load_openai_model(self, model_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Load OpenAI model."""
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
    
    def _load_phi_model(self, model_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Load Phi model."""
        model_id = f"microsoft/{model_name}"
        
        if "vision" in model_name.lower():
            processor = AutoProcessor.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            return {"model": model, "processor": processor}
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            return {"model": model, "tokenizer": tokenizer}
    
    def _load_t5_model(self, model_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Load T5 model."""
        model_id = f"google/{model_name}"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        return {"model": model, "tokenizer": tokenizer}
    
    def _load_blip_model(self, model_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Load BLIP model."""
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        
        model_id = "Salesforce/blip2-opt-2.7b"
        processor = Blip2Processor.from_pretrained(model_id)
        model = Blip2ForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        return {"model": model, "processor": processor}
    
    def clear_cache(self) -> None:
        """Clear the model cache to free up memory."""
        self.models_cache.clear()