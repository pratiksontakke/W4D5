import openai
import numpy as np
from typing import List, Union
import os
from tqdm import tqdm
import time

class OpenAIEmbedder:
    def __init__(self, api_key: str = None):
        """
        Initialize OpenAI embedder
        
        Args:
            api_key: OpenAI API key. If None, will try to get from environment variable
        """
        # Get API key from environment variable if not provided
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. Either pass it to the constructor "
                "or set the OPENAI_API_KEY environment variable."
            )
        
        openai.api_key = self.api_key
        self.model = "text-embedding-ada-002"
        
    def get_embeddings(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Get embeddings using OpenAI's text-embedding-ada-002 model
        
        Args:
            texts: Single text string or list of text strings
            
        Returns:
            numpy array of embeddings
        """
        # Convert single text to list
        if isinstance(texts, str):
            texts = [texts]
            
        embeddings = []
        # Process in batches of 100 (OpenAI's rate limit is 60 RPM)
        batch_size = 100
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating OpenAI embeddings"):
            batch = texts[i:i + batch_size]
            try:
                response = openai.Embedding.create(
                    model=self.model,
                    input=batch
                )
                batch_embeddings = [data['embedding'] for data in response['data']]
                embeddings.extend(batch_embeddings)
                
                # Sleep to respect rate limits if more batches coming
                if i + batch_size < len(texts):
                    time.sleep(0.5)  # 500ms delay between batches
                    
            except Exception as e:
                print(f"Error getting embeddings for batch {i//batch_size}: {str(e)}")
                # Return empty embeddings in case of error
                return np.zeros((len(texts), 1536))  # ada-002 produces 1536-dimensional vectors
                
        return np.array(embeddings) 