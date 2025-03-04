import fitz  # PyMuPDF
import PIL.Image
import io
import os
import logging
from typing import Dict, List, Tuple, Any
import json
from dataclasses import dataclass
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ExtractedImage:
    """Class to store extracted image data and metadata."""
    image_data: bytes
    page_number: int
    bbox: Tuple[float, float, float, float]
    surrounding_text: str
    caption: str
    image_type: str
    dpi: Tuple[int, int]

class PDFImageExtractor:
    def __init__(self, output_dir: str = "data/processed/images"):
        """Initialize the PDF image extractor.
        
        Args:
            output_dir (str): Directory to save extracted images
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def extract_images_from_pdf(self, pdf_path: str) -> Dict[str, List[ExtractedImage]]:
        """Extract images and their context from a PDF file.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            Dict[str, List[ExtractedImage]]: Dictionary mapping image IDs to their data
        """
        doc = fitz.open(pdf_path)
        images = {}
        
        for page_num, page in enumerate(doc):
            # Get text blocks for context
            text_blocks = page.get_text("blocks")
            
            # Extract images
            image_list = page.get_images(full=True)
            
            for img_idx, img in enumerate(image_list):
                try:
                    # Get image data
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    
                    # Get image position
                    bbox = page.get_image_bbox(img)
                    
                    # Find nearby text (caption and context)
                    surrounding_text, caption = self._find_image_context(text_blocks, bbox)
                    
                    # Create image identifier
                    image_id = f"page_{page_num + 1}_img_{img_idx + 1}"
                    
                    # Create ExtractedImage object
                    extracted_img = ExtractedImage(
                        image_data=image_bytes,
                        page_number=page_num + 1,
                        bbox=bbox,
                        surrounding_text=surrounding_text,
                        caption=caption,
                        image_type=image_ext,
                        dpi=self._get_image_dpi(image_bytes)
                    )
                    
                    images[image_id] = extracted_img
                    
                    # Save image and metadata
                    self._save_image_and_metadata(
                        image_id,
                        extracted_img,
                        os.path.basename(pdf_path)
                    )
                    
                except Exception as e:
                    logger.error(f"Error extracting image {img_idx} from page {page_num}: {str(e)}")
                    continue
        
        doc.close()
        return images
    
    def _find_image_context(
        self,
        text_blocks: List[Tuple],
        image_bbox: Tuple[float, float, float, float]
    ) -> Tuple[str, str]:
        """Find text surrounding an image and potential caption.
        
        Args:
            text_blocks: List of text blocks from the page
            image_bbox: Bounding box of the image
            
        Returns:
            Tuple[str, str]: (surrounding text, caption)
        """
        surrounding_text = []
        potential_captions = []
        
        img_y_mid = (image_bbox[1] + image_bbox[3]) / 2
        
        for block in text_blocks:
            block_bbox = block[:4]
            block_text = block[4]
            
            # Check if text block is near the image
            if self._is_text_near_image(block_bbox, image_bbox):
                surrounding_text.append(block_text)
                
                # Check if this might be a caption
                if self._is_likely_caption(block_bbox, image_bbox):
                    potential_captions.append(block_text)
        
        # Use the closest text block as caption if found
        caption = potential_captions[0] if potential_captions else ""
        
        return " ".join(surrounding_text), caption
    
    def _is_text_near_image(
        self,
        text_bbox: Tuple[float, float, float, float],
        image_bbox: Tuple[float, float, float, float],
        threshold: float = 50  # pixels
    ) -> bool:
        """Check if text block is near the image."""
        # Vertical distance
        v_dist = min(
            abs(text_bbox[1] - image_bbox[3]),  # text top - image bottom
            abs(text_bbox[3] - image_bbox[1])   # text bottom - image top
        )
        
        # Horizontal distance
        h_dist = min(
            abs(text_bbox[0] - image_bbox[2]),  # text left - image right
            abs(text_bbox[2] - image_bbox[0])   # text right - image left
        )
        
        return v_dist <= threshold or h_dist <= threshold
    
    def _is_likely_caption(
        self,
        text_bbox: Tuple[float, float, float, float],
        image_bbox: Tuple[float, float, float, float]
    ) -> bool:
        """Check if text block is likely to be an image caption."""
        # Check if text is below or above the image and centered
        text_center_x = (text_bbox[0] + text_bbox[2]) / 2
        image_center_x = (image_bbox[0] + image_bbox[2]) / 2
        
        is_centered = abs(text_center_x - image_center_x) < 50
        is_adjacent = (
            abs(text_bbox[1] - image_bbox[3]) < 20 or  # text immediately below image
            abs(text_bbox[3] - image_bbox[1]) < 20     # text immediately above image
        )
        
        return is_centered and is_adjacent
    
    def _get_image_dpi(self, image_bytes: bytes) -> Tuple[int, int]:
        """Get image DPI information."""
        try:
            with PIL.Image.open(io.BytesIO(image_bytes)) as img:
                return img.info.get('dpi', (72, 72))  # default to 72 DPI if not specified
        except Exception:
            return (72, 72)
    
    def _save_image_and_metadata(
        self,
        image_id: str,
        extracted_img: ExtractedImage,
        pdf_name: str
    ) -> None:
        """Save extracted image and its metadata."""
        # Create directory for this PDF if it doesn't exist
        pdf_image_dir = os.path.join(self.output_dir, pdf_name.replace('.pdf', ''))
        os.makedirs(pdf_image_dir, exist_ok=True)
        
        # Save image
        image_path = os.path.join(pdf_image_dir, f"{image_id}.{extracted_img.image_type}")
        with open(image_path, 'wb') as f:
            f.write(extracted_img.image_data)
        
        # Save metadata
        metadata = {
            'page_number': extracted_img.page_number,
            'bbox': extracted_img.bbox,
            'surrounding_text': extracted_img.surrounding_text,
            'caption': extracted_img.caption,
            'image_type': extracted_img.image_type,
            'dpi': extracted_img.dpi
        }
        
        metadata_path = os.path.join(pdf_image_dir, f"{image_id}_metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2) 