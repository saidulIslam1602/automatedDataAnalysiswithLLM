import os
import logging
from data_processing.pdf_extractor import PDFExtractor
from data_processing.text_preprocessor import TextPreprocessor
from data_processing.image_extractor import PDFImageExtractor
from models.llm_handler import LLMHandler
from models.robot_query_handler import RobotManualQueryHandler
import json
from typing import List, Dict, Any
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RobotManualAnalyzer:
    def __init__(
        self,
        pdf_dir: str = "userManual",
        processed_dir: str = "data/processed",
        model_output_dir: str = "models/fine_tuned"
    ):
        """Initialize RobotManualAnalyzer with necessary components.
        
        Args:
            pdf_dir (str): Directory containing PDF files
            processed_dir (str): Directory for processed data
            model_output_dir (str): Directory for saving fine-tuned models
        """
        self.pdf_dir = pdf_dir
        self.processed_dir = processed_dir
        self.model_output_dir = model_output_dir
        
        # Initialize components
        self.pdf_extractor = PDFExtractor(pdf_dir)
        self.image_extractor = PDFImageExtractor(os.path.join(processed_dir, "images"))
        self.text_preprocessor = TextPreprocessor()
        self.llm_handler = LLMHandler()
        self.query_handler = RobotManualQueryHandler()
        
        # Create necessary directories
        os.makedirs(processed_dir, exist_ok=True)
        os.makedirs(model_output_dir, exist_ok=True)
        
        # Storage for processed data
        self.documents: Dict[str, str] = {}
        self.processed_sentences: Dict[str, List[str]] = {}
        self.document_embeddings: Dict[str, np.ndarray] = {}
        self.image_data: Dict[str, Dict[str, Any]] = {}
        
    def extract_and_preprocess(self):
        """Extract text and images from PDFs and preprocess them."""
        logger.info("Extracting text from robot manuals...")
        self.documents = self.pdf_extractor.process_directory()
        
        logger.info("Extracting images from robot manuals...")
        for filename in os.listdir(self.pdf_dir):
            if filename.lower().endswith('.pdf'):
                pdf_path = os.path.join(self.pdf_dir, filename)
                self.image_data[filename] = self.image_extractor.extract_images_from_pdf(pdf_path)
        
        logger.info("Preprocessing extracted text...")
        for filename, text in self.documents.items():
            processed = self.text_preprocessor.process_document(text)
            self.processed_sentences[filename] = processed
            
            # Save processed text
            output_path = os.path.join(self.processed_dir, f"{os.path.splitext(filename)[0]}_processed.json")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'original_text': text,
                    'processed_sentences': processed
                }, f, indent=2)
                
    def compute_embeddings(self):
        """Compute embeddings for all processed documents."""
        logger.info("Computing document embeddings...")
        for filename, sentences in self.processed_sentences.items():
            self.document_embeddings[filename] = self.llm_handler.encode_text(sentences)
            
            # Save embeddings
            output_path = os.path.join(self.processed_dir, f"{os.path.splitext(filename)[0]}_embeddings.npy")
            np.save(output_path, self.document_embeddings[filename])
            
    def query_manuals(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Query the robot manuals with enhanced context.
        
        Args:
            query (str): User's query
            top_k (int): Number of results to return
            
        Returns:
            Dict[str, Any]: Formatted results with sections
        """
        # Enhance query with robot-specific context
        enhanced_query = self.query_handler.generate_focused_query(query)
        logger.info(f"Enhanced query: {enhanced_query}")
        
        # Perform semantic search
        raw_results = []
        for filename, embeddings in self.document_embeddings.items():
            results = self.llm_handler.semantic_search(
                enhanced_query,
                embeddings,
                self.processed_sentences[filename],
                top_k
            )
            
            for result in results:
                result['document_name'] = filename
                raw_results.append(result)
                
        # Sort by score and get top_k overall
        raw_results.sort(key=lambda x: x['score'], reverse=True)
        top_results = raw_results[:top_k]
        
        # Format results with robot manual-specific structure
        formatted_results = self.query_handler.format_search_results(top_results)
        
        # Add relevant images
        formatted_results["relevant_images"] = self._find_relevant_images(query, top_results)
        
        return formatted_results
    
    def _find_relevant_images(self, query: str, text_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find images relevant to the query and text results.
        
        Args:
            query (str): User's query
            text_results: Top text results
            
        Returns:
            List[Dict[str, Any]]: Relevant images with metadata
        """
        relevant_images = []
        
        # Get page numbers from top text results
        relevant_pages = set()
        for result in text_results:
            doc_name = result['document_name']
            if doc_name in self.image_data:
                for image_id, image_info in self.image_data[doc_name].items():
                    if image_info.page_number not in relevant_pages:
                        # Check if image context matches query
                        if (query.lower() in image_info.surrounding_text.lower() or
                            query.lower() in image_info.caption.lower()):
                            relevant_images.append({
                                'document_name': doc_name,
                                'image_id': image_id,
                                'page_number': image_info.page_number,
                                'caption': image_info.caption,
                                'context': image_info.surrounding_text
                            })
                        relevant_pages.add(image_info.page_number)
        
        return relevant_images
    
    def run_analysis(self):
        """Run the complete analysis pipeline."""
        logger.info("Starting robot manual analysis pipeline...")
        
        # Extract and preprocess text and images
        self.extract_and_preprocess()
        
        # Compute embeddings
        self.compute_embeddings()
        
        logger.info("Analysis pipeline completed successfully!")
        
def main():
    # Initialize analyzer
    analyzer = RobotManualAnalyzer()
    
    # Run analysis pipeline
    analyzer.run_analysis()
    
    # Example queries
    example_queries = [
        "What are the safety requirements for operating the robot?",
        "Show me how to perform routine maintenance on the robot",
        "What should I do if the robot encounters an error?",
        "What are the robot's technical specifications?",
        "How do I install and set up the robot?"
    ]
    
    print("\nExample Queries and Results:")
    for query in example_queries:
        print(f"\nQuery: {query}")
        results = analyzer.query_manuals(query)
        
        print("\nMain Answers:")
        for answer in results["main_answer"][:2]:  # Show top 2 main answers
            print(f"\nFrom {answer['source']}:")
            print(f"Relevance: {answer['score']:.4f}")
            print(f"Text: {answer['text']}")
            
        if results["safety_notes"]:
            print("\nRelated Safety Notes:")
            for note in results["safety_notes"][:1]:  # Show top safety note
                print(f"\nFrom {note['source']}:")
                print(f"Text: {note['text']}")
        
        if results["relevant_images"]:
            print("\nRelevant Images:")
            for img in results["relevant_images"][:2]:
                print(f"\nFrom {img['document_name']} (Page {img['page_number']}):")
                if img['caption']:
                    print(f"Caption: {img['caption']}")
                print(f"Context: {img['context']}")
        
        print(f"\nRelevant Manuals: {', '.join(results['manual_references'])}")

if __name__ == "__main__":
    main() 