"""
Evaluation framework for legal document search system.

This module implements:
1. Precision and recall metrics
2. Diversity scoring
3. Side-by-side method comparison
4. Statistical significance testing
5. Visualization utilities

The module is designed to evaluate and compare different similarity methods
using standard IR metrics and custom legal-domain specific measures.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from config import EVALUATION_CONFIG, SIMILARITY_METHODS
from core.retrieval import SearchResult
from scipy import stats
from sklearn.metrics import average_precision_score, precision_recall_curve

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Container for evaluation metrics."""

    method: str
    precision: float
    recall: float
    diversity_score: float
    avg_precision: float
    statistical_significance: Dict[str, float] = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "method": self.method,
            "precision": self.precision,
            "recall": self.recall,
            "diversity_score": self.diversity_score,
            "avg_precision": self.avg_precision,
            "statistical_significance": self.statistical_significance,
            "metadata": self.metadata,
        }


class EvaluationFramework:
    """Main evaluation framework class."""

    def __init__(
        self, relevance_threshold: float = EVALUATION_CONFIG["relevance_threshold"]
    ):
        """Initialize evaluation framework.

        Args:
            relevance_threshold: Minimum score for a document to be considered relevant
        """
        self.relevance_threshold = relevance_threshold

        # Set style for visualizations
        plt.style.use("seaborn")

    def evaluate_method(
        self,
        search_result: SearchResult,
        relevant_docs: Set[str],
        method: str,
        top_k: int = EVALUATION_CONFIG["top_k"],
    ) -> EvaluationResult:
        """Evaluate a single similarity method.

        Args:
            search_result: Search results to evaluate
            relevant_docs: Set of relevant document IDs
            method: Name of similarity method
            top_k: Number of top results to consider

        Returns:
            EvaluationResult object with computed metrics
        """
        # Get top-k results
        results = search_result.results[:top_k]
        result_ids = {doc.doc_id for doc in results}

        # Calculate basic metrics
        precision = self._calculate_precision(result_ids, relevant_docs)
        recall = self._calculate_recall(result_ids, relevant_docs)
        diversity = self._calculate_diversity(results)

        # Calculate average precision
        scores = [doc.metadata.get("score", 0) for doc in search_result.results]
        relevance = [
            1 if doc.doc_id in relevant_docs else 0 for doc in search_result.results
        ]
        avg_precision = average_precision_score(relevance, scores)

        # Create result object
        result = EvaluationResult(
            method=method,
            precision=precision,
            recall=recall,
            diversity_score=diversity,
            avg_precision=avg_precision,
            metadata={
                "num_results": len(results),
                "query": search_result.query,
                "threshold": self.relevance_threshold,
            },
        )

        return result

    def compare_methods(
        self,
        results: Dict[str, SearchResult],
        relevant_docs: Set[str],
        top_k: int = EVALUATION_CONFIG["top_k"],
    ) -> Dict[str, EvaluationResult]:
        """Compare multiple similarity methods.

        Args:
            results: Dictionary of method names to search results
            relevant_docs: Set of relevant document IDs
            top_k: Number of top results to consider

        Returns:
            Dictionary of method names to evaluation results
        """
        eval_results = {}

        # Evaluate each method
        for method, result in results.items():
            eval_result = self.evaluate_method(result, relevant_docs, method, top_k)
            eval_results[method] = eval_result

        # Compute statistical significance
        self._compute_significance(eval_results)

        return eval_results

    def visualize_comparison(
        self,
        eval_results: Dict[str, EvaluationResult],
        output_path: Optional[Path] = None,
    ) -> None:
        """Create visualization comparing methods.

        Args:
            eval_results: Dictionary of evaluation results
            output_path: Optional path to save visualization
        """
        # Convert results to DataFrame
        data = []
        for method, result in eval_results.items():
            data.append(
                {
                    "Method": method,
                    "Precision": result.precision,
                    "Recall": result.recall,
                    "Diversity": result.diversity_score,
                    "MAP": result.avg_precision,
                }
            )
        df = pd.DataFrame(data)

        # Create subplot grid
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Method Comparison", fontsize=16)

        # Plot metrics
        metrics = ["Precision", "Recall", "Diversity", "MAP"]
        for ax, metric in zip(axes.flat, metrics):
            sns.barplot(data=df, x="Method", y=metric, ax=ax)
            ax.set_title(f"{metric} by Method")
            ax.tick_params(axis="x", rotation=45)

        plt.tight_layout()

        # Save or show
        if output_path:
            plt.savefig(output_path)
        else:
            plt.show()

        plt.close()

    def _calculate_precision(
        self, result_ids: Set[str], relevant_ids: Set[str]
    ) -> float:
        """Calculate precision metric.

        Args:
            result_ids: Set of retrieved document IDs
            relevant_ids: Set of relevant document IDs

        Returns:
            Precision score
        """
        if not result_ids:
            return 0.0

        relevant_retrieved = len(result_ids.intersection(relevant_ids))
        return relevant_retrieved / len(result_ids)

    def _calculate_recall(self, result_ids: Set[str], relevant_ids: Set[str]) -> float:
        """Calculate recall metric.

        Args:
            result_ids: Set of retrieved document IDs
            relevant_ids: Set of relevant document IDs

        Returns:
            Recall score
        """
        if not relevant_ids:
            return 0.0

        relevant_retrieved = len(result_ids.intersection(relevant_ids))
        return relevant_retrieved / len(relevant_ids)

    def _calculate_diversity(self, results: List[SearchResult]) -> float:
        """Calculate diversity score.

        Args:
            results: List of search results

        Returns:
            Diversity score
        """
        if not results:
            return 0.0

        # Extract document types
        doc_types = [doc.doc_type for doc in results]

        # Calculate type diversity (unique types / total types)
        type_diversity = len(set(doc_types)) / len(doc_types)

        # Calculate score spread
        scores = [doc.metadata.get("score", 0) for doc in results]
        score_spread = np.std(scores) if scores else 0

        # Combine metrics (weighted average)
        diversity_score = 0.7 * type_diversity + 0.3 * score_spread

        return diversity_score

    def _compute_significance(self, eval_results: Dict[str, EvaluationResult]) -> None:
        """Compute statistical significance between methods.

        Args:
            eval_results: Dictionary of evaluation results
        """
        methods = list(eval_results.keys())

        # Compare each pair of methods
        for i, method1 in enumerate(methods):
            for method2 in methods[i + 1 :]:
                # Get precision scores
                scores1 = [eval_results[method1].precision]
                scores2 = [eval_results[method2].precision]

                # Perform t-test
                t_stat, p_value = stats.ttest_ind(scores1, scores2)

                # Store results
                eval_results[method1].statistical_significance[method2] = p_value
                eval_results[method2].statistical_significance[method1] = p_value


if __name__ == "__main__":
    # Example usage
    evaluator = EvaluationFramework()

    # Create sample search results
    from core.retrieval import DocumentInfo

    def create_sample_result(method: str, scores: List[float]) -> SearchResult:
        """Helper to create sample results."""
        docs = []
        for i, score in enumerate(scores):
            doc = DocumentInfo(
                doc_id=f"doc_{i}",
                title=f"Document {i}",
                content=f"Sample content {i}",
                doc_type=["income_tax_act", "gst_act"][i % 2],
                metadata={"score": score},
            )
            docs.append(doc)

        return SearchResult(
            query="sample query",
            results=docs,
            scores={method: np.mean(scores)},
            metadata={"method": method},
        )

    # Create sample results for different methods
    search_results = {
        "cosine": create_sample_result("cosine", [0.9, 0.8, 0.7, 0.6, 0.5]),
        "euclidean": create_sample_result("euclidean", [0.85, 0.75, 0.65, 0.55, 0.45]),
        "mmr": create_sample_result("mmr", [0.95, 0.7, 0.6, 0.5, 0.4]),
    }

    # Define relevant documents
    relevant_docs = {f"doc_{i}" for i in range(3)}  # First 3 docs are relevant

    try:
        # Compare methods
        eval_results = evaluator.compare_methods(search_results, relevant_docs)

        # Print results
        print("\nEvaluation Results:")
        for method, result in eval_results.items():
            print(f"\n{method.upper()}:")
            print(f"Precision: {result.precision:.3f}")
            print(f"Recall: {result.recall:.3f}")
            print(f"Diversity: {result.diversity_score:.3f}")
            print(f"MAP: {result.avg_precision:.3f}")

            if result.statistical_significance:
                print("\nStatistical Significance (p-values):")
                for other_method, p_value in result.statistical_significance.items():
                    print(f"vs {other_method}: {p_value:.3f}")

        # Create visualization
        evaluator.visualize_comparison(eval_results)

    except Exception as e:
        print(f"Error: {str(e)}")
