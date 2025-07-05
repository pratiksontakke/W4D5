import joblib
import numpy as np
from core.data_loader import load_ag_news_data
from core.embedders import Word2VecEmbedder
from core.classifier import UnifiedClassifier
import config

def main():
    # Set random seed for reproducibility
    np.random.seed(config.RANDOM_SEED)
    
    # Load and prepare data
    train_texts, train_labels, label_map = load_ag_news_data(
        samples_per_category=config.SAMPLES_PER_CATEGORY,
        seed=config.RANDOM_SEED
    )
    
    # Initialize Word2Vec embedder and classifier
    print("\nInitializing Word2Vec embedder and classifier...")
    embedder = Word2VecEmbedder()  # Using default downloaded model
    classifier = UnifiedClassifier(embedder=embedder, categories=config.CATEGORIES)
    
    # Train the classifier
    print("\nTraining classifier...")
    metrics = classifier.train(train_texts, train_labels)
    
    # Print training metrics
    print("\nTraining Metrics:")
    for category in config.CATEGORIES:
        print(f"\n{category}:")
        if metrics[category]['support'] > 0:
            print(f"Precision: {metrics[category]['precision']:.3f}")
            print(f"Recall: {metrics[category]['recall']:.3f}")
            print(f"F1-score: {metrics[category]['f1-score']:.3f}")
            print(f"Support: {metrics[category]['support']}")
        else:
            print("No training data available yet")
    
    # Save the trained model
    model_path = config.MODELS_DIR / "word2vec_classifier.joblib"
    joblib.dump(classifier, model_path)
    print(f"\nSaved model to {model_path}")
    
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