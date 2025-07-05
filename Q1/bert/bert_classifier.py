from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import numpy as np
from typing import List, Dict, Any, Union
from bert_embedder import BertEmbedder

class BertClassifier:
    def __init__(self, categories: List[str]):
        """
        Initialize BERT classifier
        
        Args:
            categories: List of category names for classification
        """
        self.embedder = BertEmbedder()
        self.classifier = LogisticRegression(
            max_iter=1000,
            multi_class='multinomial'
        )
        self.categories = categories
        
    def train(self, texts: List[str], labels: List[str]) -> Dict[str, float]:
        """
        Train the classifier on text data
        
        Args:
            texts: List of text samples
            labels: List of corresponding labels
            
        Returns:
            Dictionary containing training metrics
        """
        # Get embeddings for all texts
        print("Generating BERT embeddings...")
        embeddings = self.embedder.get_embeddings(texts)
        
        # Train classifier
        print("Training classifier...")
        self.classifier.fit(embeddings, labels)
        
        # Get training metrics
        predictions = self.classifier.predict(embeddings)
        report = classification_report(labels, predictions, target_names=self.categories, output_dict=True)
        
        return report
    
    def predict(self, texts: Union[str, List[str]]) -> Dict[str, Any]:
        """
        Make predictions on new texts
        
        Args:
            texts: Single text or list of texts to classify
            
        Returns:
            Dictionary containing predictions and confidence scores
        """
        # Get embeddings
        embeddings = self.embedder.get_embeddings(texts)
        
        # Get predictions and probabilities
        predictions = self.classifier.predict(embeddings)
        probabilities = self.classifier.predict_proba(embeddings)
        
        # Format results
        if isinstance(texts, str):
            result = {
                'prediction': predictions[0],
                'confidence': float(np.max(probabilities[0])),
                'probabilities': {
                    cat: float(prob) 
                    for cat, prob in zip(self.categories, probabilities[0])
                }
            }
        else:
            result = {
                'predictions': predictions.tolist(),
                'confidences': np.max(probabilities, axis=1).tolist(),
                'probabilities': [
                    {cat: float(prob) for cat, prob in zip(self.categories, probs)}
                    for probs in probabilities
                ]
            }
            
        return result 