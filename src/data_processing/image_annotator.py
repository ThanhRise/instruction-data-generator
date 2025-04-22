from typing import Dict, List, Any, Optional, Union
import logging
from pathlib import Path
import torch
from PIL import Image
import numpy as np
from transformers import (
    BlipProcessor, 
    BlipForConditionalGeneration,
    AutoModelForObjectDetection,
    DetrImageProcessor
)

logger = logging.getLogger(__name__)

class ImageAnnotator:
    """Annotates images with captions and visual information."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize image annotator with specified models.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Initialize caption generation model
        try:
            model_config = config["models"]["image_processing"]
            if "blip2" in model_config["caption_model"].lower():
                self.caption_processor = BlipProcessor.from_pretrained(
                    model_config["caption_model"]
                )
                self.caption_model = BlipForConditionalGeneration.from_pretrained(
                    model_config["caption_model"],
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
            else:
                # Initialize Phi-vision or other model
                from transformers import AutoModelForCausalLM, AutoTokenizer
                self.caption_processor = AutoTokenizer.from_pretrained(
                    "microsoft/phi-3.5-vision-instruct",
                    trust_remote_code=True
                )
                self.caption_model = AutoModelForCausalLM.from_pretrained(
                    "microsoft/phi-3.5-vision-instruct",
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
        except Exception as e:
            logger.error(f"Error loading caption model: {e}")
            self.caption_model = None
            self.caption_processor = None
        
        # Initialize object detection model
        try:
            self.object_processor = DetrImageProcessor.from_pretrained(
                model_config["object_detection"]
            )
            self.object_model = AutoModelForObjectDetection.from_pretrained(
                model_config["object_detection"],
                torch_dtype=torch.float16,
                device_map="auto"
            )
        except Exception as e:
            logger.error(f"Error loading object detection model: {e}")
            self.object_model = None
            self.object_processor = None
    
    def annotate_images(
        self,
        images: List[Dict[str, Any]],
        output_dir: Optional[Path] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate annotations for a list of images.
        
        Args:
            images: List of image data dictionaries
            output_dir: Optional directory to save processed images
            
        Returns:
            List of annotated image data
        """
        annotated_images = []
        
        for img_data in images:
            try:
                # Get image and metadata
                image = img_data.get("image")
                if not image:
                    continue
                    
                source = img_data.get("source", "unknown")
                img_type = img_data.get("type", "image")
                doc_type = img_data.get("document_type", "")
                
                # Generate annotations
                annotations = self._generate_annotations(
                    image,
                    img_type,
                    doc_type
                )
                
                # Add annotations to image data
                annotated_data = {
                    "source": source,
                    "type": img_type,
                    "document_type": doc_type,
                    "annotation": annotations["caption"],
                    "objects": annotations["objects"],
                    "visual_elements": annotations["visual_elements"],
                    "confidence": annotations["confidence"]
                }
                
                # Save processed image if output directory provided
                if output_dir:
                    output_dir.mkdir(parents=True, exist_ok=True)
                    output_path = output_dir / f"{Path(source).stem}_processed.png"
                    self._save_processed_image(image, annotations, output_path)
                    annotated_data["processed_path"] = str(output_path)
                
                annotated_images.append(annotated_data)
                
            except Exception as e:
                logger.error(f"Error annotating image {img_data.get('source', 'unknown')}: {e}")
                continue
        
        return annotated_images
    
    def _generate_annotations(
        self,
        image: Image.Image,
        img_type: str,
        doc_type: str = ""
    ) -> Dict[str, Any]:
        """Generate comprehensive annotations for an image."""
        annotations = {
            "caption": "",
            "objects": [],
            "visual_elements": [],
            "confidence": 0.0
        }
        
        try:
            # Generate caption based on image type
            if img_type == "pdf_page":
                caption = self._generate_page_caption(image, doc_type)
            elif img_type in ["slide_content", "ppt_image"]:
                caption = self._generate_slide_caption(image, doc_type)
            elif img_type in ["excel_image", "chart"]:
                caption = self._generate_chart_caption(image)
            else:
                caption = self._generate_basic_caption(image)
            
            annotations["caption"] = caption
            
            # Detect objects and visual elements
            if self.object_model:
                objects = self._detect_objects(image)
                annotations["objects"] = objects
                
                # Extract visual elements based on objects
                visual_elements = self._analyze_visual_elements(objects, img_type)
                annotations["visual_elements"] = visual_elements
            
            # Set confidence based on detection results
            if annotations["objects"]:
                annotations["confidence"] = float(np.mean([obj["score"] for obj in annotations["objects"]]))
            else:
                annotations["confidence"] = 0.7  # Default confidence for caption-only
            
            return annotations
            
        except Exception as e:
            logger.error(f"Error generating annotations: {e}")
            return annotations
    
    def _generate_basic_caption(self, image: Image.Image) -> str:
        """Generate basic image caption."""
        try:
            if "blip2" in str(self.caption_model.__class__):
                # Use BLIP-2 model
                inputs = self.caption_processor(image, return_tensors="pt").to(
                    self.caption_model.device
                )
                outputs = self.caption_model.generate(
                    **inputs,
                    max_new_tokens=100,
                    num_beams=5,
                    min_length=10,
                    repetition_penalty=1.5
                )
                caption = self.caption_processor.decode(outputs[0], skip_special_tokens=True)
            else:
                # Use Phi-vision model
                prompt = "Describe this image in detail:"
                inputs = self.caption_processor.process_images(
                    prompt,
                    image,
                    max_length=100,
                    temperature=0.7
                )
                outputs = self.caption_model.generate(**inputs)
                caption = self.caption_processor.decode(outputs[0], skip_special_tokens=True)
            
            return caption.strip()
            
        except Exception as e:
            logger.error(f"Error generating basic caption: {e}")
            return "Error generating caption"
    
    def _generate_page_caption(self, image: Image.Image, doc_type: str) -> str:
        """Generate caption for document page image."""
        try:
            prompt = f"This is a page from a {doc_type} document. Describe the visual layout and content:"
            
            if "blip2" in str(self.caption_model.__class__):
                inputs = self.caption_processor(
                    image, 
                    text=prompt,
                    return_tensors="pt"
                ).to(self.caption_model.device)
                
                outputs = self.caption_model.generate(
                    **inputs,
                    max_new_tokens=150,
                    num_beams=5,
                    min_length=20,
                    repetition_penalty=1.5
                )
                caption = self.caption_processor.decode(outputs[0], skip_special_tokens=True)
            else:
                inputs = self.caption_processor.process_images(
                    prompt,
                    image,
                    max_length=150,
                    temperature=0.7
                )
                outputs = self.caption_model.generate(**inputs)
                caption = self.caption_processor.decode(outputs[0], skip_special_tokens=True)
            
            return caption.strip()
            
        except Exception as e:
            logger.error(f"Error generating page caption: {e}")
            return f"A page from a {doc_type} document"
    
    def _generate_slide_caption(self, image: Image.Image, doc_type: str) -> str:
        """Generate caption for presentation slide."""
        try:
            prompt = "This is a presentation slide. Describe the key content and visual elements:"
            
            if "blip2" in str(self.caption_model.__class__):
                inputs = self.caption_processor(
                    image,
                    text=prompt,
                    return_tensors="pt"
                ).to(self.caption_model.device)
                
                outputs = self.caption_model.generate(
                    **inputs,
                    max_new_tokens=150,
                    num_beams=5,
                    min_length=20,
                    repetition_penalty=1.5
                )
                caption = self.caption_processor.decode(outputs[0], skip_special_tokens=True)
            else:
                inputs = self.caption_processor.process_images(
                    prompt,
                    image,
                    max_length=150,
                    temperature=0.7
                )
                outputs = self.caption_model.generate(**inputs)
                caption = self.caption_processor.decode(outputs[0], skip_special_tokens=True)
            
            return caption.strip()
            
        except Exception as e:
            logger.error(f"Error generating slide caption: {e}")
            return "A presentation slide"
    
    def _generate_chart_caption(self, image: Image.Image) -> str:
        """Generate caption for chart or graph."""
        try:
            prompt = "This is a chart or graph. Describe what it shows and any key trends or patterns:"
            
            if "blip2" in str(self.caption_model.__class__):
                inputs = self.caption_processor(
                    image,
                    text=prompt,
                    return_tensors="pt"
                ).to(self.caption_model.device)
                
                outputs = self.caption_model.generate(
                    **inputs,
                    max_new_tokens=150,
                    num_beams=5,
                    min_length=20,
                    repetition_penalty=1.5
                )
                caption = self.caption_processor.decode(outputs[0], skip_special_tokens=True)
            else:
                inputs = self.caption_processor.process_images(
                    prompt,
                    image,
                    max_length=150,
                    temperature=0.7
                )
                outputs = self.caption_model.generate(**inputs)
                caption = self.caption_processor.decode(outputs[0], skip_special_tokens=True)
            
            return caption.strip()
            
        except Exception as e:
            logger.error(f"Error generating chart caption: {e}")
            return "A chart or graph"
    
    def _detect_objects(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Detect objects and visual elements in image."""
        try:
            inputs = self.object_processor(image, return_tensors="pt").to(
                self.object_model.device
            )
            outputs = self.object_model(**inputs)
            
            # Convert outputs to COCO format
            target_sizes = torch.tensor([image.size[::-1]])
            results = self.object_processor.post_process_object_detection(
                outputs,
                target_sizes=target_sizes,
                threshold=0.5
            )[0]
            
            objects = []
            for score, label, box in zip(
                results["scores"],
                results["labels"],
                results["boxes"]
            ):
                objects.append({
                    "label": self.object_model.config.id2label[label.item()],
                    "score": score.item(),
                    "box": box.tolist()
                })
            
            return objects
            
        except Exception as e:
            logger.error(f"Error detecting objects: {e}")
            return []
    
    def _analyze_visual_elements(
        self,
        objects: List[Dict[str, Any]],
        img_type: str
    ) -> List[Dict[str, Any]]:
        """Analyze detected objects to identify visual elements."""
        visual_elements = []
        
        try:
            # Group objects by type
            element_groups = {}
            for obj in objects:
                label = obj["label"]
                if label not in element_groups:
                    element_groups[label] = []
                element_groups[label].append(obj)
            
            # Identify visual elements based on image type
            if img_type in ["chart", "excel_image"]:
                # Look for chart elements
                chart_elements = ["line", "bar", "point", "axis", "legend"]
                for label, group in element_groups.items():
                    if any(elem in label.lower() for elem in chart_elements):
                        visual_elements.append({
                            "type": "chart_element",
                            "name": label,
                            "count": len(group),
                            "confidence": float(np.mean([obj["score"] for obj in group]))
                        })
                        
            elif img_type in ["slide_content", "ppt_image"]:
                # Look for slide elements
                slide_elements = ["text", "title", "bullet", "image", "shape"]
                for label, group in element_groups.items():
                    if any(elem in label.lower() for elem in slide_elements):
                        visual_elements.append({
                            "type": "slide_element",
                            "name": label,
                            "count": len(group),
                            "confidence": float(np.mean([obj["score"] for obj in group]))
                        })
                        
            else:
                # General visual elements
                for label, group in element_groups.items():
                    visual_elements.append({
                        "type": "visual_element",
                        "name": label,
                        "count": len(group),
                        "confidence": float(np.mean([obj["score"] for obj in group]))
                    })
            
            return visual_elements
            
        except Exception as e:
            logger.error(f"Error analyzing visual elements: {e}")
            return []
    
    def _save_processed_image(
        self,
        image: Image.Image,
        annotations: Dict[str, Any],
        output_path: Path
    ) -> None:
        """Save processed image with visualized annotations."""
        try:
            # Create copy of image for drawing
            from PIL import ImageDraw
            
            processed_img = image.copy()
            draw = ImageDraw.Draw(processed_img)
            
            # Draw detected objects
            for obj in annotations.get("objects", []):
                box = obj["box"]
                label = obj["label"]
                score = obj["score"]
                
                # Draw bounding box
                draw.rectangle(box, outline="red", width=2)
                
                # Draw label
                label_text = f"{label}: {score:.2f}"
                draw.text((box[0], box[1] - 10), label_text, fill="red")
            
            # Save image
            processed_img.save(output_path)
            
        except Exception as e:
            logger.error(f"Error saving processed image: {e}")
            # Save original image if processing fails
            image.save(output_path)