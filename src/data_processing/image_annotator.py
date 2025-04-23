from typing import Dict, List, Any, Optional
import logging
import torch
from PIL import Image
import numpy as np
from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
    AutoModelForImageToText,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    pipeline
)
from sentence_transformers import SentenceTransformer
import pytesseract
import cv2

logger = logging.getLogger(__name__)

class ImageAnnotator:
    """Enhanced image content extraction and analysis."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.image_config = config["agent"]["document_processing"]["image_processing"]
        
        # Initialize components based on configuration
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize vision models and processors."""
        try:
            # Object detection model
            if self.image_config.get("use_object_detection", True):
                model_name = self.image_config.get(
                    "object_detection_model",
                    "microsoft/faster-rcnn-resnet50-fpn"
                )
                self.obj_processor = AutoImageProcessor.from_pretrained(model_name)
                self.obj_model = AutoModelForObjectDetection.from_pretrained(model_name)
            
            # Image captioning model
            if self.image_config.get("use_image_captioning", True):
                model_name = self.image_config.get(
                    "caption_model",
                    "Salesforce/blip2-opt-2.7b"
                )
                self.caption_pipeline = pipeline(
                    "image-to-text",
                    model=model_name,
                    device=0 if torch.cuda.is_available() else -1
                )
            
            # Visual question answering
            if self.image_config.get("use_visual_qa", True):
                self.vqa_pipeline = pipeline(
                    "visual-question-answering",
                    model=self.image_config.get(
                        "vqa_model",
                        "dandelin/vilt-b32-finetuned-vqa"
                    )
                )
                
            # Scene understanding model
            if self.image_config.get("use_scene_understanding", True):
                self.scene_model = pipeline(
                    "image-to-text",
                    model=self.image_config.get(
                        "scene_model",
                        "microsoft/git-large-coco"
                    )
                )
            
            # OCR configuration
            if self.image_config.get("use_ocr", True):
                self.ocr_config = {
                    "lang": self.image_config.get("ocr_language", "eng"),
                    "config": f"--psm {self.image_config.get('psm', 3)} --oem {self.image_config.get('oem', 3)}"
                }
            
        except Exception as e:
            logger.error(f"Failed to initialize image processing components: {e}")
            raise
    
    def process_image(self, image: Image.Image) -> Dict[str, Any]:
        """
        Process an image using multiple vision models.
        
        Args:
            image: PIL Image to process
            
        Returns:
            Dictionary containing extracted information
        """
        result = {
            "ocr_text": "",
            "caption": "",
            "context": {
                "objects": [],
                "scene_description": "",
                "visual_qa": []
            }
        }
        
        try:
            # Perform OCR if enabled
            if self.image_config.get("use_ocr", True):
                result["ocr_text"] = self._extract_text(image)
            
            # Generate image caption
            if self.image_config.get("use_image_captioning", True):
                result["caption"] = self._generate_caption(image)
            
            # Detect objects
            if self.image_config.get("use_object_detection", True):
                result["context"]["objects"] = self._detect_objects(image)
            
            # Generate scene description
            if self.image_config.get("use_scene_understanding", True):
                result["context"]["scene_description"] = self._analyze_scene(image)
            
            # Perform visual QA
            if self.image_config.get("use_visual_qa", True):
                result["context"]["visual_qa"] = self._perform_visual_qa(image)
            
            return result
            
        except Exception as e:
            logger.error(f"Image processing failed: {e}")
            return result
    
    def _extract_text(self, image: Image.Image) -> str:
        """Extract text from image using OCR."""
        try:
            # Convert to RGB if needed
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            # Preprocess image for better OCR
            img_array = np.array(image)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            processed = cv2.threshold(
                gray, 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )[1]
            
            # Perform OCR
            text = pytesseract.image_to_string(
                processed,
                lang=self.ocr_config["lang"],
                config=self.ocr_config["config"]
            )
            
            return text.strip()
            
        except Exception as e:
            logger.warning(f"OCR failed: {e}")
            return ""
    
    def _generate_caption(self, image: Image.Image) -> str:
        """Generate descriptive caption for the image."""
        try:
            result = self.caption_pipeline(image)
            if result and isinstance(result, list) and len(result) > 0:
                return result[0].get("generated_text", "").strip()
            return ""
            
        except Exception as e:
            logger.warning(f"Image captioning failed: {e}")
            return ""
    
    def _detect_objects(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Detect objects in the image with confidence scores."""
        try:
            # Prepare image for model
            inputs = self.obj_processor(images=image, return_tensors="pt")
            
            # Run inference
            with torch.no_grad():
                outputs = self.obj_model(**inputs)
            
            # Process results
            processed_results = self.obj_processor.post_process_object_detection(
                outputs,
                threshold=self.image_config.get("detection_threshold", 0.5)
            )[0]
            
            objects = []
            for score, label, box in zip(
                processed_results["scores"],
                processed_results["labels"],
                processed_results["boxes"]
            ):
                objects.append({
                    "label": self.obj_model.config.id2label[label.item()],
                    "confidence": score.item(),
                    "box": box.tolist()
                })
            
            return objects
            
        except Exception as e:
            logger.warning(f"Object detection failed: {e}")
            return []
    
    def _analyze_scene(self, image: Image.Image) -> str:
        """Generate detailed scene description."""
        try:
            result = self.scene_model(image)
            if result and isinstance(result, list) and len(result) > 0:
                return result[0].get("generated_text", "").strip()
            return ""
            
        except Exception as e:
            logger.warning(f"Scene analysis failed: {e}")
            return ""
    
    def _perform_visual_qa(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Perform visual question answering on the image."""
        try:
            # Standard questions for scene understanding
            questions = [
                "What is happening in this image?",
                "What is the main subject of this image?",
                "What is the setting or location?",
                "Are there any people in this image?",
                "What objects are most prominent?",
                "What is the overall mood or atmosphere?"
            ]
            
            results = []
            for question in questions:
                answer = self.vqa_pipeline(
                    image=image,
                    question=question,
                    top_k=1
                )
                if answer and isinstance(answer, list) and len(answer) > 0:
                    results.append({
                        "question": question,
                        "answer": answer[0].get("answer", "").strip(),
                        "confidence": answer[0].get("score", 0.0)
                    })
            
            return results
            
        except Exception as e:
            logger.warning(f"Visual QA failed: {e}")
            return []
    
    def combine_results(self, results: List[Dict[str, Any]]) -> str:
        """
        Combine multiple image processing results into a coherent text representation.
        
        Args:
            results: List of image processing results
            
        Returns:
            Combined text representation
        """
        combined_sections = []
        
        for result in results:
            sections = []
            
            # Add OCR text if available
            if result["ocr_text"]:
                sections.append(f"[OCR Text: {result['ocr_text']}]")
            
            # Add image caption
            if result["caption"]:
                sections.append(f"[Image Caption: {result['caption']}]")
            
            # Add object detections
            if result["context"]["objects"]:
                obj_text = ", ".join(
                    f"{obj['label']} ({obj['confidence']:.2f})"
                    for obj in result["context"]["objects"]
                )
                sections.append(f"[Detected Objects: {obj_text}]")
            
            # Add scene description
            if result["context"]["scene_description"]:
                sections.append(
                    f"[Scene Description: {result['context']['scene_description']}]"
                )
            
            # Add visual QA results
            if result["context"]["visual_qa"]:
                qa_text = " | ".join(
                    f"{qa['question']} -> {qa['answer']}"
                    for qa in result["context"]["visual_qa"]
                )
                sections.append(f"[Visual Analysis: {qa_text}]")
            
            if sections:
                combined_sections.append("\n".join(sections))
        
        return "\n\n".join(combined_sections)