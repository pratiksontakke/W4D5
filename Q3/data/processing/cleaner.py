"""
Text cleaning and normalization utilities for sales conversation data.

This module provides a comprehensive set of text cleaning functions specifically
designed for sales conversation transcripts, including:
- Text normalization
- Noise removal
- Special character handling
- Domain-specific cleaning rules
"""

import re
from typing import List, Optional, Set
import unicodedata

import nltk
import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from config.config import LANGCHAIN_CONFIG

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Load spaCy model
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    spacy.cli.download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

class TextCleaner:
    """Text cleaning and normalization for sales conversation data."""
    
    def __init__(self, 
                 remove_stopwords: bool = False,
                 remove_numbers: bool = False,
                 remove_punctuation: bool = False,
                 lowercase: bool = True,
                 custom_stopwords: Optional[Set[str]] = None):
        """Initialize the text cleaner with specified options.
        
        Args:
            remove_stopwords: Whether to remove common stopwords
            remove_numbers: Whether to remove numeric values
            remove_punctuation: Whether to remove punctuation
            lowercase: Whether to convert text to lowercase
            custom_stopwords: Additional domain-specific stopwords to remove
        """
        self.remove_stopwords = remove_stopwords
        self.remove_numbers = remove_numbers
        self.remove_punctuation = remove_punctuation
        self.lowercase = lowercase
        
        # Initialize stopwords
        self.stopwords = set(stopwords.words('english'))
        if custom_stopwords:
            self.stopwords.update(custom_stopwords)
            
        # Common sales-specific patterns
        self.money_pattern = re.compile(r'\$\d+(?:,\d{3})*(?:\.\d{2})?')
        self.email_pattern = re.compile(r'\S+@\S+\.\S+')
        self.phone_pattern = re.compile(r'\+?1?[-\s.]?\(?\d{3}\)?[-\s.]?\d{3}[-\s.]?\d{4}|\+?1?[-\s.]?\d{10}|\+\d{1,3}[-\s.]?\d{9,15}')
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        
    def normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace in text.
        
        Args:
            text: Input text
            
        Returns:
            Text with normalized whitespace
        """
        # Replace multiple whitespace characters with a single space
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def remove_special_characters(self, text: str, keep_punctuation: bool = True) -> str:
        """Remove or replace special characters.
        
        Args:
            text: Input text
            keep_punctuation: Whether to keep basic punctuation
            
        Returns:
            Cleaned text
        """
        # Normalize unicode characters
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
        
        if keep_punctuation:
            # Keep basic punctuation but remove other special characters
            text = re.sub(r'[^\w\s.,!?-]', ' ', text)
        else:
            # Remove all special characters including punctuation
            text = re.sub(r'[^\w\s]', ' ', text)
            
        return self.normalize_whitespace(text)
    
    def normalize_entities(self, text: str) -> str:
        """Normalize common entities like money, emails, phones, and URLs.
        
        Args:
            text: Input text
            
        Returns:
            Text with normalized entities
        """
        # Replace entities with placeholders
        text = self.money_pattern.sub('[MONEY]', text)
        text = self.email_pattern.sub('[EMAIL]', text)
        text = self.phone_pattern.sub('[PHONE]', text)
        text = self.url_pattern.sub('[URL]', text)
        return text
    
    def clean_text(self, text: str) -> str:
        """Apply all cleaning steps to the text.
        
        Args:
            text: Input text
            
        Returns:
            Fully cleaned and normalized text
        """
        if not text or not isinstance(text, str):
            raise ValueError("Input must be a non-empty string")
        
        # Convert to lowercase if specified
        if self.lowercase:
            text = text.lower()
        
        # Normalize entities
        text = self.normalize_entities(text)
        
        # Remove special characters
        text = self.remove_special_characters(text, keep_punctuation=not self.remove_punctuation)
        
        # Tokenize for word-level operations
        words = word_tokenize(text)
        
        # Apply filters
        if self.remove_numbers:
            words = [word for word in words if not word.isdigit()]
            
        if self.remove_stopwords:
            words = [word for word in words if word.lower() not in self.stopwords]
        
        # Rejoin words
        text = ' '.join(words)
        
        return self.normalize_whitespace(text)
    
    def clean_batch(self, texts: List[str]) -> List[str]:
        """Clean a batch of texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of cleaned texts
        """
        return [self.clean_text(text) for text in texts]
    
    def validate_text(self, text: str) -> bool:
        """Validate if the text meets minimum quality requirements.
        
        Args:
            text: Input text
            
        Returns:
            True if text is valid, False otherwise
        """
        if not text or not isinstance(text, str):
            return False
            
        # Check minimum length (configurable)
        min_length = LANGCHAIN_CONFIG.get("min_text_length", 10)
        if len(text.split()) < min_length:
            return False
            
        # Check for meaningful content (not just special characters)
        cleaned = self.remove_special_characters(text, keep_punctuation=False)
        if not cleaned.strip():
            return False
            
        return True

if __name__ == "__main__":
    # Example usage
    cleaner = TextCleaner(
        remove_stopwords=True,
        remove_numbers=False,
        remove_punctuation=False,
        lowercase=True,
        custom_stopwords={'um', 'uh', 'er'}
    )
    
    # Test with a sample sales conversation text
    sample_text = """
    Hi there! My name is John Smith and I'm calling from TechCorp.
    Our product costs $1,299.99 and you can reach me at john@techcorp.com
    or +1-123-456-7890. Check out our website at https://techcorp.com!
    Um... would you be interested in learning more?
    """
    
    print("Original text:")
    print(sample_text)
    print("\nCleaned text:")
    print(cleaner.clean_text(sample_text))
    
    # Test batch processing
    texts = [
        "First sample text with $100",
        "Second sample text with john@email.com",
        "Third sample text with https://example.com"
    ]
    print("\nBatch cleaning results:")
    for original, cleaned in zip(texts, cleaner.clean_batch(texts)):
        print(f"\nOriginal: {original}")
        print(f"Cleaned:  {cleaned}") 