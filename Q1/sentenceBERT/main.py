from sentence_bert_classifier import SentenceBertClassifier
from datasets import load_dataset
from tqdm import tqdm

def main():
    # Define our categories
    categories = ['Tech', 'Finance', 'Healthcare', 'Sports', 'Politics', 'Entertainment']
    
    # Initialize classifier with specific model
    classifier = SentenceBertClassifier(
        categories=categories,
        model_name='all-MiniLM-L6-v2'  # Using the specified model
    )
    
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
    
    print("\nNote: Training with available categories from AG News dataset:")
    print("- Politics (mapped from World news)")
    print("- Sports")
    print("- Finance (mapped from Business)")
    print("- Tech (mapped from Sci/Tech)")
    print("\nCategories 'Healthcare' and 'Entertainment' will be added later with additional data.")
    
    # Prepare training data (using a small subset for demonstration)
    train_texts = dataset['train']['text'][:1000]  # Using first 1000 samples
    train_labels = [label_map[label] for label in dataset['train']['label'][:1000]]
    
    # Train the classifier
    print("\nTraining classifier...")
    metrics = classifier.train(train_texts, train_labels)
    
    # Print training metrics
    print("\nTraining Metrics:")
    for category in categories:
        print(f"\n{category}:")
        if metrics[category]['support'] > 0:
            print(f"Precision: {metrics[category]['precision']:.3f}")
            print(f"Recall: {metrics[category]['recall']:.3f}")
            print(f"F1-score: {metrics[category]['f1-score']:.3f}")
            print(f"Support: {metrics[category]['support']}")
        else:
            print("No training data available yet")
    
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
        # Sort probabilities by value for better readability
        sorted_probs = sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True)
        for cat, prob in sorted_probs:
            if prob > 0:  # Only show non-zero probabilities
                print(f"  {cat}: {prob:.3f}")

if __name__ == "__main__":
    main() 