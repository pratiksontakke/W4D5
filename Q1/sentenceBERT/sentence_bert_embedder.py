from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union

class SentenceBertEmbedder:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize Sentence-BERT embedder
        
        Args:
            model_name: Name of the Sentence-BERT model to use
        """
        self.model = SentenceTransformer(model_name)
        
    def get_embeddings(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Get sentence embeddings directly using Sentence-BERT
        
        Args:
            texts: Single text string or list of text strings
            
        Returns:
            numpy array of embeddings
        """
        # Convert single text to list
        if isinstance(texts, str):
            texts = [texts]
            
        # Get embeddings
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2 normalization
        )
        
        return embeddings 