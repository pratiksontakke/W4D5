"""
Evaluation pipeline for measuring search system performance.

Computes precision, recall, and diversity metrics for different similarity methods.
"""
import logging
from typing import Dict, List

import numpy as np
from sklearn.metrics import precision_score, recall_score

from config import EVALUATION_CONFIG
from core.evaluation import calculate_diversity, calculate_precision, calculate_recall

logger = logging.getLogger(__name__)


def evaluate_similarity_methods(
    query_results: Dict[str, List[Dict]], relevant_docs: List[str]
) -> Dict[str, float]:
    """
    Evaluate performance of different similarity methods.

    Breaks long line into multiple lines for better readability.
    """
    results = {}
    for method, docs in query_results.items():
        precision = calculate_precision(docs, relevant_docs)
        recall = calculate_recall(docs, relevant_docs)
        diversity = calculate_diversity(docs)
        results[method] = {
            "precision": precision,
            "recall": recall,
            "diversity": diversity,
        }
    return results


# ... rest of the file ...
