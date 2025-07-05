from openai_classifier import OpenAIClassifier
from datasets import load_dataset
from tqdm import tqdm
import os
from collections import defaultdict
import random
import numpy as np

def main():
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Define our categories
    categories = ['Tech', 'Finance', 'Healthcare', 'Sports', 'Politics', 'Entertainment']
    
    # Initialize classifier
    classifier = OpenAIClassifier(categories=categories)
    
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
    
    # Prepare balanced training data
    samples_per_category = 25  # We'll get 25 samples for each available category
    
    # First, collect all available texts for each category
    all_texts_by_category = defaultdict(list)
    for i in range(len(dataset['train'])):
        label = dataset['train'][i]['label']
        category = label_map[label]
        all_texts_by_category[category].append(dataset['train'][i]['text'])
    
    # Then sample equally from each category
    train_texts = []
    train_labels = []
    
    print("\nSampling training data:")
    for category in label_map.values():
        # Randomly sample texts for this category
        category_texts = random.sample(all_texts_by_category[category], samples_per_category)
        train_texts.extend(category_texts)
        train_labels.extend([category] * samples_per_category)
        print(f"- {category}: {samples_per_category} samples")
    
    # Shuffle the training data while preserving pairs
    combined = list(zip(train_texts, train_labels))
    random.shuffle(combined)
    train_texts, train_labels = map(list, zip(*combined))  # Convert back to lists
    
    # Verify distribution
    print("\nVerifying category distribution:")
    category_counts = defaultdict(int)
    for label in train_labels:
        category_counts[label] += 1
    for category, count in category_counts.items():
        print(f"- {category}: {count} samples")
    
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