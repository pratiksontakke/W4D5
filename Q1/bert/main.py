from bert_classifier import BertClassifier
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

def main():
    # Define our categories
    categories = ['Tech', 'Finance', 'Healthcare', 'Sports', 'Politics', 'Entertainment']
    
    # Initialize classifier
    classifier = BertClassifier(categories=categories)
    
    # Load AG News dataset
    print("Loading AG News dataset...")
    dataset = load_dataset("ag_news")
    
    # Map AG News labels to our categories (simplified mapping for demonstration)
    label_map = {
        0: 'Politics',    # World news mapped to Politics
        1: 'Sports',      # Sports
        2: 'Finance',     # Business mapped to Finance
        3: 'Tech'         # Sci/Tech mapped to Tech
    }
    
    # Prepare training data (using a small subset for demonstration)
    train_texts = dataset['train']['text'][:1000]  # Using first 1000 samples
    train_labels = [label_map[label] for label in dataset['train']['label'][:1000]]
    
    # Train the classifier
    print("\nTraining classifier...")
    metrics = classifier.train(train_texts, train_labels)
    
    # Print training metrics
    print("\nTraining Metrics:")
    for category in categories:
        if category in metrics:
            print(f"\n{category}:")
            print(f"Precision: {metrics[category]['precision']:.3f}")
            print(f"Recall: {metrics[category]['recall']:.3f}")
            print(f"F1-score: {metrics[category]['f1-score']:.3f}")
    
    # Example predictions
    test_texts = [
        "Apple announces new iPhone with revolutionary AI capabilities",
        "Manchester United wins dramatic match against Liverpool",
        "New healthcare bill passes in Senate with bipartisan support",
        "Stock market reaches all-time high as tech sector surges",
    ]
    
    print("\nMaking predictions on test texts...")
    for text in test_texts:
        result = classifier.predict(text)
        print(f"\nText: {text}")
        print(f"Predicted Category: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print("Probabilities:")
        for cat, prob in result['probabilities'].items():
            print(f"  {cat}: {prob:.3f}")

if __name__ == "__main__":
    main() 