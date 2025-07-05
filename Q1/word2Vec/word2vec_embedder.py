import numpy as np
from gensim.models import KeyedVectors
import nltk
from nltk.tokenize import word_tokenize
from typing import List, Union
import os

class Word2VecEmbedder:
    def __init__(self, model_path: str = None):
        """
        Initialize Word2Vec embedder with pre-trained model
        
        Args:
            model_path: Path to pre-trained Word2Vec/GloVe model
                       If None, will download a small model for testing
        """
        # Download required NLTK data
        nltk.download('punkt', quiet=True)
        
        # Load or download model
        if model_path and os.path.exists(model_path):
            self.model = KeyedVectors.load_word2vec_format(model_path, binary=True)
        else:
            # Download a small pre-trained model for testing
            import gensim.downloader as api
            self.model = api.load('glove-wiki-gigaword-100')  # 100-dimensional GloVe embeddings
            
        self.vector_size = self.model.vector_size
        
    def _preprocess_text(self, text: str) -> List[str]:
        """
        Tokenize and preprocess text
        
        Args:
            text: Input text string
            
        Returns:
            List of tokens
        """
        # Convert to lower case and tokenize
        tokens = word_tokenize(text.lower())
        # Filter out tokens not in vocabulary
        tokens = [token for token in tokens if token in self.model.key_to_index]
        return tokens
        
    def get_embeddings(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Get document embeddings by averaging word vectors
        
        Args:
            texts: Single text string or list of text strings
            
        Returns:
            numpy array of document embeddings
        """
        # Convert single text to list
        if isinstance(texts, str):
            texts = [texts]
            
        # Get embeddings for each text
        embeddings = []
        for text in texts:
            tokens = self._preprocess_text(text)
            if not tokens:
                # If no valid tokens, return zero vector
                embeddings.append(np.zeros(self.vector_size))
                continue
                
            # Get word vectors and average them
            word_vectors = [self.model[token] for token in tokens]
            doc_vector = np.mean(word_vectors, axis=0)
            embeddings.append(doc_vector)
            
        return np.array(embeddings) 