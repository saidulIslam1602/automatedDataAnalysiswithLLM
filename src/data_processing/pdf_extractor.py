import PyPDF2
import os
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFExtractor:
    def __init__(self, pdf_dir: str):
        """Initialize PDFExtractor with directory containing PDF files.
        
        Args:
            pdf_dir (str): Path to directory containing PDF files
        """
        self.pdf_dir = pdf_dir
        self.extracted_data: Dict[str, str] = {}
        
    def extract_from_file(self, pdf_path: str) -> Optional[str]:
        """Extract text from a single PDF file.
        
        Args:
            pdf_path (str): Path to PDF file
            
        Returns:
            str: Extracted text or None if extraction fails
        """
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {str(e)}")
            return None
    
    def process_directory(self) -> Dict[str, str]:
        """Process all PDF files in the specified directory.
        
        Returns:
            Dict[str, str]: Dictionary mapping filenames to extracted text
        """
        for filename in os.listdir(self.pdf_dir):
            if filename.lower().endswith('.pdf'):
                pdf_path = os.path.join(self.pdf_dir, filename)
                logger.info(f"Processing {filename}")
                
                extracted_text = self.extract_from_file(pdf_path)
                if extracted_text:
                    self.extracted_data[filename] = extracted_text
                    
        return self.extracted_data
    
    def save_extracted_text(self, output_dir: str) -> None:
        """Save extracted text to individual text files.
        
        Args:
            output_dir (str): Directory to save text files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for filename, text in self.extracted_data.items():
            output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.txt")
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(text)
            logger.info(f"Saved extracted text to {output_path}") 