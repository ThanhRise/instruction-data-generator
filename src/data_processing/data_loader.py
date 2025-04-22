from typing import Dict, List, Any, Optional, Union
import logging
from pathlib import Path
import json
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from .document_processor import DocumentProcessor
from ..utils.helpers import setup_logging

logger = logging.getLogger(__name__)

class DataLoader:
    """Handles loading and preprocessing of input data from various sources."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize data loader with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.input_config = config["agent"]["input_processing"]
        self.doc_processor = DocumentProcessor(config)
        
        # Set up input/output paths
        self.base_path = Path(config["paths"]["base_dir"])
        self.input_dir = self.base_path / "data/input"
        self.processed_dir = self.base_path / "data/processed"
        self.output_dir = self.base_path / "data/output"
        
        # Create necessary directories
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize document tracking
        self.processed_docs = {}
        self._load_processing_state()
    
    def load_documents(
        self,
        input_paths: Optional[List[Union[str, Path]]] = None,
        batch_size: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Load and process documents from specified paths or input directory.
        
        Args:
            input_paths: Optional list of specific file paths to process
            batch_size: Optional limit on number of documents to process
            
        Returns:
            List of processed document data
        """
        # Get files to process
        if input_paths:
            files_to_process = [Path(p) for p in input_paths]
        else:
            files_to_process = self._get_input_files()
        
        if batch_size:
            files_to_process = files_to_process[:batch_size]
        
        processed_data = []
        
        # Process documents in parallel
        with ThreadPoolExecutor(max_workers=self.input_config.get("max_workers", 4)) as executor:
            future_to_file = {
                executor.submit(self._process_document, file_path): file_path
                for file_path in files_to_process
            }
            
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    doc_data = future.result()
                    if doc_data:
                        processed_data.append(doc_data)
                        self._update_processing_state(file_path, True)
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
                    self._update_processing_state(file_path, False, str(e))
        
        return processed_data
    
    def _get_input_files(self) -> List[Path]:
        """Get list of input files to process."""
        files = []
        
        # Collect files from text directory
        text_dir = self.input_dir / "text"
        if text_dir.exists():
            files.extend(
                path for path in text_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in {
                    ".txt", ".md", ".doc", ".docx", ".pdf",
                    ".ppt", ".pptx", ".xls", ".xlsx"
                }
            )
        
        # Collect files from images directory
        image_dir = self.input_dir / "images"
        if image_dir.exists():
            files.extend(
                path for path in image_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in {
                    ".png", ".jpg", ".jpeg", ".gif", ".bmp"
                }
            )
        
        # Filter out already processed files unless reprocessing is enabled
        if not self.input_config.get("reprocess_existing", False):
            files = [f for f in files if not self._is_processed(f)]
        
        return sorted(files)
    
    def _process_document(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Process a single document."""
        try:
            # Process based on file type
            if file_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".bmp"}:
                # Handle image files
                return self._process_image_file(file_path)
            else:
                # Handle document files
                return self._process_doc_file(file_path)
                
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {e}")
            return None
    
    def _process_image_file(self, file_path: Path) -> Dict[str, Any]:
        """Process an image file."""
        from PIL import Image
        
        try:
            # Load image
            with Image.open(file_path) as img:
                # Create document data
                doc_data = {
                    "source": str(file_path),
                    "type": "image",
                    "format": file_path.suffix.lower()[1:],
                    "image": img.copy(),
                    "metadata": {
                        "size": img.size,
                        "mode": img.mode
                    }
                }
                
                return doc_data
                
        except Exception as e:
            logger.error(f"Error processing image {file_path}: {e}")
            return None
    
    def _process_doc_file(self, file_path: Path) -> Dict[str, Any]:
        """Process a document file."""
        try:
            # Extract content using document processor
            doc_content = self.doc_processor.process_document(file_path)
            
            # Create document data
            doc_data = {
                "source": str(file_path),
                "type": "document",
                "format": file_path.suffix.lower()[1:],
                "content": doc_content["text"],
                "images": doc_content["images"],
                "relationships": doc_content["relationships"]
            }
            
            return doc_data
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {e}")
            return None
    
    def _load_processing_state(self) -> None:
        """Load document processing state from disk."""
        state_file = self.processed_dir / "processing_state.json"
        if state_file.exists():
            try:
                with open(state_file, "r") as f:
                    self.processed_docs = json.load(f)
            except Exception as e:
                logger.error(f"Error loading processing state: {e}")
                self.processed_docs = {}
    
    def _update_processing_state(
        self,
        file_path: Path,
        success: bool,
        error: Optional[str] = None
    ) -> None:
        """Update document processing state."""
        state_file = self.processed_dir / "processing_state.json"
        
        # Update state
        self.processed_docs[str(file_path)] = {
            "success": success,
            "timestamp": str(datetime.now()),
            "error": error
        }
        
        # Save to disk
        try:
            with open(state_file, "w") as f:
                json.dump(self.processed_docs, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving processing state: {e}")
    
    def _is_processed(self, file_path: Path) -> bool:
        """Check if a file has been successfully processed."""
        doc_state = self.processed_docs.get(str(file_path), {})
        return doc_state.get("success", False)
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get document processing statistics."""
        stats = {
            "total_documents": len(self.processed_docs),
            "successful": sum(
                1 for doc in self.processed_docs.values()
                if doc.get("success", False)
            ),
            "failed": sum(
                1 for doc in self.processed_docs.values()
                if not doc.get("success", False)
            ),
            "formats": {}
        }
        
        # Count by format
        for file_path in self.processed_docs:
            file_format = Path(file_path).suffix.lower()[1:]
            if file_format not in stats["formats"]:
                stats["formats"][file_format] = 0
            stats["formats"][file_format] += 1
        
        return stats