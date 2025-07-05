from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from typing import List, Union

class BertEmbedder:
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