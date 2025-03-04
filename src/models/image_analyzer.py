from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import torch
from typing import Dict, List, Any, Tuple
import logging
from pathlib import Path
import numpy as np
from torchvision import transforms
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ImageAnalysis:
    """Container for image analysis results."""
    content_type: str  # e.g., "diagram", "photo", "illustration"
    description: str
    confidence: float
    features: np.ndarray
    technical_details: Dict[str, Any]

class DeepImageAnalyzer:
    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """Initialize the deep learning-based image analyzer.
        
        Args:
            model_name: Name of the pretrained ViT model
            device: Device to run the model on
        """
        self.device = device
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.model = ViTForImageClassification.from_pretrained(model_name).to(device)
        
        # Custom transforms for technical diagrams
        self.diagram_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(3),  # Convert to 3-channel grayscale
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
    def analyze_image(self, image: Image.Image) -> ImageAnalysis:
        """Analyze an image using the ViT model.
        
        Args:
            image: PIL Image to analyze
            
        Returns:
            ImageAnalysis: Analysis results
        """
        # Prepare image
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        # Get model predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            
        # Get features from the last hidden state
        features = outputs.hidden_states[-1].mean(dim=1).cpu().numpy()
        
        # Get prediction
        pred_id = probs.argmax().item()
        confidence = probs.max().item()
        
        # Analyze image type and content
        content_type = self._determine_content_type(image)
        description = self._generate_description(
            self.model.config.id2label[pred_id],
            content_type,
            features
        )
        
        # Extract technical details
        technical_details = self._extract_technical_details(image, features)
        
        return ImageAnalysis(
            content_type=content_type,
            description=description,
            confidence=confidence,
            features=features,
            technical_details=technical_details
        )
    
    def _determine_content_type(self, image: Image.Image) -> str:
        """Determine the type of technical content in the image."""
        # Convert to grayscale for analysis
        gray_image = image.convert('L')
        
        # Analyze image characteristics
        histogram = gray_image.histogram()
        edges = self._detect_edges(gray_image)
        
        # Heuristics for content type determination
        if self._is_diagram(histogram, edges):
            return "technical_diagram"
        elif self._is_schematic(histogram, edges):
            return "schematic"
        elif self._is_illustration(histogram, edges):
            return "illustration"
        else:
            return "photo"
    
    def _detect_edges(self, image: Image.Image) -> np.ndarray:
        """Detect edges using image gradients."""
        img_array = np.array(image)
        gradient_x = np.gradient(img_array, axis=1)
        gradient_y = np.gradient(img_array, axis=0)
        return np.sqrt(gradient_x**2 + gradient_y**2)
    
    def _is_diagram(self, histogram: List[int], edges: np.ndarray) -> bool:
        """Check if image is likely a technical diagram."""
        edge_density = np.mean(edges > np.percentile(edges, 90))
        histogram_peaks = self._count_peaks(histogram)
        return edge_density > 0.1 and histogram_peaks < 5
    
    def _is_schematic(self, histogram: List[int], edges: np.ndarray) -> bool:
        """Check if image is likely a schematic."""
        edge_density = np.mean(edges > np.percentile(edges, 90))
        histogram_peaks = self._count_peaks(histogram)
        return edge_density > 0.15 and histogram_peaks < 3
    
    def _is_illustration(self, histogram: List[int], edges: np.ndarray) -> bool:
        """Check if image is likely an illustration."""
        edge_density = np.mean(edges > np.percentile(edges, 90))
        histogram_peaks = self._count_peaks(histogram)
        return edge_density < 0.1 and histogram_peaks > 5
    
    def _count_peaks(self, histogram: List[int], threshold: float = 0.1) -> int:
        """Count significant peaks in histogram."""
        hist_array = np.array(histogram)
        peak_indices = (hist_array[1:-1] > hist_array[:-2]) & (hist_array[1:-1] > hist_array[2:])
        significant_peaks = hist_array[1:-1][peak_indices] > np.max(hist_array) * threshold
        return np.sum(significant_peaks)
    
    def _generate_description(
        self,
        label: str,
        content_type: str,
        features: np.ndarray
    ) -> str:
        """Generate a detailed description of the image content."""
        # Combine model prediction with content type analysis
        description_parts = [f"This image appears to be a {content_type}"]
        
        if content_type == "technical_diagram":
            description_parts.append("containing technical specifications or layouts")
        elif content_type == "schematic":
            description_parts.append("showing component relationships or system architecture")
        elif content_type == "illustration":
            description_parts.append("demonstrating concepts or procedures")
            
        description_parts.append(f"classified as '{label}' by the model")
        
        return " ".join(description_parts)
    
    def _extract_technical_details(
        self,
        image: Image.Image,
        features: np.ndarray
    ) -> Dict[str, Any]:
        """Extract technical details from the image."""
        return {
            "dimensions": image.size,
            "aspect_ratio": image.size[0] / image.size[1],
            "color_mode": image.mode,
            "feature_statistics": {
                "mean": float(np.mean(features)),
                "std": float(np.std(features)),
                "min": float(np.min(features)),
                "max": float(np.max(features))
            }
        } 