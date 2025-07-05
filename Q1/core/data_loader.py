from datasets import load_dataset
from collections import defaultdict
import random
from typing import Tuple, List, Dict

def load_ag_news_data(samples_per_category: int = 25, seed: int = 42) -> Tuple[List[str], List[str], Dict[int, str]]:
    """
    Load and prepare AG News dataset
    
    Args:
        samples_per_category: Number of samples to get for each category
        seed: Random seed for reproducibility
        
    Returns:
        Tuple containing:
        - List of training texts
        - List of training labels
        - Label mapping dictionary
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Define label mapping
    label_map = {
        0: 'Politics',    # World news mapped to Politics
        1: 'Sports',      # Sports
        2: 'Finance',     # Business mapped to Finance
        3: 'Tech'         # Sci/Tech mapped to Tech
    }
    
    # Load AG News dataset
    print("Loading AG News dataset...")
    dataset = load_dataset("ag_news")
    
    print("\nNote: Training with available categories from AG News dataset:")
    print("- Politics (mapped from World news)")
    print("- Sports")
    print("- Finance (mapped from Business)")
    print("- Tech (mapped from Sci/Tech)")
    print("\nCategories 'Healthcare' and 'Entertainment' will be added later with additional data.")
    
    # Collect all available texts for each category
    all_texts_by_category = defaultdict(list)
    for i in range(len(dataset['train'])):
        label = dataset['train'][i]['label']
        category = label_map[label]
        all_texts_by_category[category].append(dataset['train'][i]['text'])
    
    # Sample equally from each category
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
    train_texts, train_labels = map(list, zip(*combined))
    
    # Verify distribution
    print("\nVerifying category distribution:")
    category_counts = defaultdict(int)
    for label in train_labels:
        category_counts[label] += 1
    for category, count in category_counts.items():
        print(f"- {category}: {count} samples")
        
    return train_texts, train_labels, label_map 