from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import numpy as np
from typing import List, Dict, Any, Union
from .embedders import Embedder

class UnifiedClassifier:
    def __init__(self, embedder: Embedder, categories: List[str]):
        """
        Initialize classifier with an embedder
        
        Args:
            embedder: Instance of an Embedder class
            categories: List of category names for classification
        """
        self.embedder = embedder
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
        print("Generating embeddings...")
        embeddings = self.embedder.get_embeddings(texts)
        
        # Train classifier
        print("Training classifier...")
        self.classifier.fit(embeddings, labels)
        
        # Get training metrics
        predictions = self.classifier.predict(embeddings)
        
        # Get unique labels actually present in the data
        unique_labels = sorted(list(set(labels)))
        
        # Generate classification report with only available categories
        report = classification_report(labels, predictions, target_names=unique_labels, output_dict=True)
        
        # Add empty metrics for missing categories
        full_report = {}
        for category in self.categories:
            if category in report:
                full_report[category] = report[category]
            else:
                full_report[category] = {
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1-score': 0.0,
                    'support': 0
                }
        
        return full_report
    
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
            # Create probability dictionary with all categories
            probs = {cat: 0.0 for cat in self.categories}
            # Update probabilities for available categories
            for cat, prob in zip(self.classifier.classes_, probabilities[0]):
                probs[cat] = float(prob)
            
            result = {
                'prediction': predictions[0],
                'confidence': float(np.max(probabilities[0])),
                'probabilities': probs
            }
        else:
            # Handle batch predictions
            all_probs = []
            for probs in probabilities:
                cat_probs = {cat: 0.0 for cat in self.categories}
                for cat, prob in zip(self.classifier.classes_, probs):
                    cat_probs[cat] = float(prob)
                all_probs.append(cat_probs)
            
            result = {
                'predictions': predictions.tolist(),
                'confidences': np.max(probabilities, axis=1).tolist(),
                'probabilities': all_probs
            }
            
        return result 