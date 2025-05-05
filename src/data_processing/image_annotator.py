from typing import Dict, List, Any, Optional
import logging
import torch
from PIL import Image
import numpy as np
from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    pipeline
)
from sentence_transformers import SentenceTransformer
import pytesseract
import cv2

logger = logging.getLogger(__name__)

class ImageAnnotator:
    """Enhanced image content extraction and analysis with efficient GPU management."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.image_config = config["agent"]["document_processing"]["image_processing"]
        self.device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
        self.batch_size = self.image_config.get("batch_size", 4)
        
        # Initialize components with device management
        self._initialize_components()
        
    def _initialize_components(self):
        """Initialize vision models and processors with device management."""
        try:
            # Object detection - lightweight model
            if self.image_config.get("use_object_detection", True):
                model_name = self.image_config.get(
                    "object_detection_model",
                    "facebook/detr-resnet-50"
                )
                self.obj_processor = AutoImageProcessor.from_pretrained(model_name)
                self.obj_model = AutoModelForObjectDetection.from_pretrained(
                    model_name,
                    device_map={"": self.device},
                    torch_dtype=torch.float16
                )
            
            # Image captioning - medium size model
            if self.image_config.get("use_image_captioning", True):
                self.caption_pipeline = pipeline(
                    "image-to-text",
                    model=self.image_config.get(
                        "caption_model",
                        "Salesforce/blip2-opt-2.7b"
                    ),
                    device_map={"": self.device},
                    torch_dtype=torch.float16,
                    model_kwargs={"load_in_8bit": True}
                )
            
            # Visual QA - lightweight model
            if self.image_config.get("use_visual_qa", True):
                self.vqa_pipeline = pipeline(
                    "visual-question-answering",
                    model=self.image_config.get(
                        "vqa_model",
                        "dandelin/vilt-b32-finetuned-vqa"
                    ),
                    device_map={"": self.device},
                    torch_dtype=torch.float16
                )
            
            # Scene understanding - medium size model
            if self.image_config.get("use_scene_understanding", True):
                self.scene_model = pipeline(
                    "image-to-text",
                    model=self.image_config.get(
                        "scene_model",
                        "microsoft/git-large-coco"
                    ),
                    device_map={"": self.device},
                    torch_dtype=torch.float16,
                    model_kwargs={"load_in_8bit": True}
                )
            
        except Exception as e:
            logger.error(f"Failed to initialize image processing components: {e}")
            raise

    def process_batch(self, images: List[Image.Image]) -> List[Dict[str, Any]]:
        """Process a batch of images efficiently."""
        results = []
        
        try:
            # Process images in batches
            for i in range(0, len(images), self.batch_size):
                batch = images[i:i + self.batch_size]
                batch_results = []
                
                # Run OCR in parallel if enabled
                if self.image_config.get("use_ocr", True):
                    ocr_texts = self._batch_extract_text(batch)
                else:
                    ocr_texts = [""] * len(batch)
                
                # Run object detection in batch
                if self.image_config.get("use_object_detection", True):
                    objects_batch = self._batch_detect_objects(batch)
                else:
                    objects_batch = [[] for _ in batch]
                
                # Generate captions in batch
                if self.image_config.get("use_image_captioning", True):
                    captions = self._batch_generate_captions(batch)
                else:
                    captions = [""] * len(batch)
                
                # Process scene descriptions in batch
                if self.image_config.get("use_scene_understanding", True):
                    scenes = self._batch_analyze_scenes(batch)
                else:
                    scenes = [""] * len(batch)
                
                # Run visual QA in batch
                if self.image_config.get("use_visual_qa", True):
                    qa_results = self._batch_perform_visual_qa(batch)
                else:
                    qa_results = [[] for _ in batch]
                
                # Combine results
                for ocr, objects, caption, scene, qa in zip(
                    ocr_texts, objects_batch, captions, scenes, qa_results
                ):
                    batch_results.append({
                        "ocr_text": ocr,
                        "caption": caption,
                        "context": {
                            "objects": objects,
                            "scene_description": scene,
                            "visual_qa": qa
                        }
                    })
                
                results.extend(batch_results)
                
                # Clear CUDA cache after each batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            raise
        
        return results

    def process_image(self, image: "Image.Image") -> Dict[str, Any]:
        """
        Process a single image and return extracted information.
        
        Args:
            image: PIL Image to process
            
        Returns:
            Dictionary containing extracted information
        """
        # Process the single image as a batch of one
        results = self.process_batch([image])
        
        # Return the first (and only) result
        return results[0] if results else {
            "ocr_text": "",
            "caption": "",
            "context": {
                "objects": [],
                "scene_description": "",
                "visual_qa": []
            }
        }

    def _batch_extract_text(self, images: List[Image.Image]) -> List[str]:
        """Extract text from a batch of images using OCR."""
        results = []
        for image in images:
            try:
                # Convert to RGB if needed
                if image.mode != "RGB":
                    image = image.convert("RGB")
                
                # Preprocess image
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
                results.append(text.strip())
            except Exception as e:
                logger.warning(f"OCR failed for an image: {e}")
                results.append("")
        
        return results

    def _batch_detect_objects(self, images: List[Image.Image]) -> List[List[Dict[str, Any]]]:
        """Detect objects in a batch of images."""
        try:
            # Prepare batch input
            inputs = self.obj_processor(images=images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Run inference
            with torch.no_grad():
                outputs = self.obj_model(**inputs)
            
            # Process results
            batch_results = []
            for i in range(len(images)):
                processed = self.obj_processor.post_process_object_detection(
                    outputs,
                    target_sizes=[images[i].size[::-1]],
                    threshold=self.image_config.get("detection_threshold", 0.5)
                )[0]
                
                objects = []
                for score, label, box in zip(
                    processed["scores"],
                    processed["labels"],
                    processed["boxes"]
                ):
                    objects.append({
                        "label": self.obj_model.config.id2label[label.item()],
                        "confidence": score.item(),
                        "box": box.tolist()
                    })
                batch_results.append(objects)
            
            return batch_results
            
        except Exception as e:
            logger.warning(f"Object detection failed: {e}")
            return [[] for _ in images]

    def _batch_generate_captions(self, images: List[Image.Image]) -> List[str]:
        """Generate captions for a batch of images."""
        try:
            results = self.caption_pipeline(
                images,
                batch_size=self.batch_size,
                max_new_tokens=self.image_config.get("max_caption_length", 50)
            )
            return [
                result[0]["generated_text"].strip() 
                if result else "" 
                for result in results
            ]
        except Exception as e:
            logger.warning(f"Image captioning failed: {e}")
            return ["" for _ in images]

    def _batch_analyze_scenes(self, images: List[Image.Image]) -> List[str]:
        """Generate scene descriptions for a batch of images."""
        try:
            results = self.scene_model(
                images,
                batch_size=self.batch_size,
                max_new_tokens=self.image_config.get("max_scene_length", 100)
            )
            return [
                result[0]["generated_text"].strip() 
                if result else "" 
                for result in results
            ]
        except Exception as e:
            logger.warning(f"Scene analysis failed: {e}")
            return ["" for _ in images]

    def _batch_perform_visual_qa(self, images: List[Image.Image]) -> List[List[Dict[str, Any]]]:
        """Perform visual QA on a batch of images."""
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
            
            batch_results = []
            for image in images:
                image_qa = []
                # Process each question for the image
                qa_inputs = [
                    {"image": image, "question": q} 
                    for q in questions
                ]
                answers = self.vqa_pipeline(
                    qa_inputs,
                    batch_size=len(questions),
                    top_k=1
                )
                
                for q, a in zip(questions, answers):
                    if isinstance(a, list) and len(a) > 0:
                        image_qa.append({
                            "question": q,
                            "answer": a[0]["answer"],
                            "confidence": a[0]["score"]
                        })
                batch_results.append(image_qa)
            
            return batch_results
            
        except Exception as e:
            logger.warning(f"Visual QA failed: {e}")
            return [[] for _ in images]

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