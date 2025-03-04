import re
import nltk
from typing import List, Optional
import logging
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextPreprocessor:
    def __init__(self):
        """Initialize TextPreprocessor with necessary NLTK components."""
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def clean_text(self, text: str) -> str:
        """Clean text by removing special characters and normalizing whitespace.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Cleaned text
        """
        # Remove special characters and digits
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', ' ', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        return text.lower()
    
    def segment_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences.
        
        Args:
            text (str): Input text
            
        Returns:
            List[str]: List of sentences
        """
        return sent_tokenize(text)
    
    def preprocess_sentence(self, sentence: str, remove_stopwords: bool = True) -> str:
        """Preprocess a single sentence with optional stopword removal and lemmatization.
        
        Args:
            sentence (str): Input sentence
            remove_stopwords (bool): Whether to remove stopwords
            
        Returns:
            str: Preprocessed sentence
        """
        # Tokenize
        words = word_tokenize(sentence)
        
        # Remove stopwords if requested
        if remove_stopwords:
            words = [w for w in words if w.lower() not in self.stop_words]
        
        # Lemmatize
        words = [self.lemmatizer.lemmatize(w) for w in words]
        
        return ' '.join(words)
    
    def process_document(self, text: str, remove_stopwords: bool = True) -> List[str]:
        """Process an entire document.
        
        Args:
            text (str): Input document text
            remove_stopwords (bool): Whether to remove stopwords
            
        Returns:
            List[str]: List of preprocessed sentences
        """
        # Clean the text
        cleaned_text = self.clean_text(text)
        
        # Split into sentences
        sentences = self.segment_into_sentences(cleaned_text)
        
        # Process each sentence
        processed_sentences = [
            self.preprocess_sentence(sent, remove_stopwords)
            for sent in sentences
        ]
        
        return processed_sentences 