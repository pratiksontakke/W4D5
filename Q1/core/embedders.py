from abc import ABC, abstractmethod
import openai
import numpy as np
from typing import List, Union
import os
from tqdm import tqdm
import time
from transformers import AutoTokenizer, AutoModel
import torch
from sentence_transformers import SentenceTransformer
from gensim.models import KeyedVectors
import nltk
from nltk.tokenize import word_tokenize

class Embedder(ABC):
    @abstractmethod
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        pass

class OpenAIEmbedder(Embedder):
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

class BertEmbedder(Embedder):
    def __init__(self, model_name: str = "bert-base-uncased", max_length: int = 512):
        """
        Initialize BERT embedder with specified model
        
        Args:
            model_name: Name of the BERT model to use
            max_length: Maximum sequence length for tokenization
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.max_length = max_length
        
        # Move model to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
    def get_embeddings(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Get BERT embeddings for input texts using [CLS] token
        
        Args:
            texts: Single text string or list of text strings
            
        Returns:
            numpy array of embeddings
        """
        # Convert single text to list
        if isinstance(texts, str):
            texts = [texts]
            
        # Tokenize texts
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Move inputs to same device as model
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        
        # Get BERT embeddings
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
        # Get [CLS] token embeddings (first token of each sequence)
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        return embeddings 

class SentenceBertEmbedder(Embedder):
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

class Word2VecEmbedder(Embedder):
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