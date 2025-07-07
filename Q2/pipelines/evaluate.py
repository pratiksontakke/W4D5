"""
Evaluation Pipeline for Indian Legal Document Search System.

This module implements the evaluation framework for comparing different similarity methods
in legal document retrieval. It includes:
1. Evaluation dataset creation
2. Metrics calculation (precision, recall, diversity)
3. Comparison framework
4. Report generation
5. Performance visualization
6. A/B testing capabilities
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.metrics import precision_recall_curve
from tqdm import tqdm

from config import (
    ARTIFACTS_DIR,
    DATA_DIR,
    EVALUATION_CONFIG,
    LEGAL_DOCUMENT_TYPES,
    SIMILARITY_METHODS,
)
from core.data_loader import DataLoader
from core.evaluation import (
    calculate_diversity_score,
    calculate_precision_at_k,
    calculate_recall_at_k,
)
from core.retrieval import DocumentRetriever
from core.similarity import (
    CosineSimCalculator,
    EuclideanSimCalculator,
    HybridSimCalculator,
    MMRCalculator,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class EvaluationPipeline:
    """Pipeline for evaluating and comparing different similarity methods."""

    def __init__(
        self,
        data_loader: DataLoader,
        document_retriever: DocumentRetriever,
        evaluation_queries: Optional[List[str]] = None,
        top_k: int = EVALUATION_CONFIG["top_k"],
        relevance_threshold: float = EVALUATION_CONFIG["relevance_threshold"],
    ):
        """
        Initialize the evaluation pipeline.

        Args:
            data_loader: Instance of DataLoader for accessing test data
            document_retriever: Instance of DocumentRetriever for similarity search
            evaluation_queries: Optional list of test queries (uses config if None)
            top_k: Number of top results to consider for metrics
            relevance_threshold: Minimum similarity score for relevance
        """
        self.data_loader = data_loader
        self.document_retriever = document_retriever
        self.evaluation_queries = (
            evaluation_queries or EVALUATION_CONFIG["test_queries"]
        )
        self.top_k = top_k
        self.relevance_threshold = relevance_threshold

        # Initialize similarity calculators
        self.similarity_calculators = {
            "cosine": CosineSimCalculator(),
            "euclidean": EuclideanSimCalculator(),
            "mmr": MMRCalculator(
                lambda_param=SIMILARITY_METHODS["mmr"]["lambda_param"],
                threshold=SIMILARITY_METHODS["mmr"]["threshold"],
            ),
            "hybrid": HybridSimCalculator(
                cosine_weight=SIMILARITY_METHODS["hybrid"]["cosine_weight"],
                entity_weight=SIMILARITY_METHODS["hybrid"]["entity_weight"],
            ),
        }

        # Create results directory
        self.results_dir = ARTIFACTS_DIR / "evaluation_results"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_cache = {}

    def build_evaluation_dataset(self) -> Dict[str, List[Dict]]:
        """
        Create evaluation dataset with ground truth annotations.

        Returns:
            Dictionary mapping queries to relevant documents with annotations
        """
        logger.info("Building evaluation dataset...")

        eval_dataset = {}
        for query in tqdm(self.evaluation_queries, desc="Processing queries"):
            # Get relevant documents for each legal document type
            relevant_docs = []
            for doc_type in LEGAL_DOCUMENT_TYPES:
                docs = self.data_loader.get_documents_by_type(doc_type)
                # Get ground truth relevance using basic keyword matching
                # In production, this should be replaced with human annotations
                relevant = [
                    {
                        "doc_id": doc["id"],
                        "relevance_score": self._calculate_basic_relevance(
                            query, doc["text"]
                        ),
                    }
                    for doc in docs
                    if self._calculate_basic_relevance(query, doc["text"])
                    >= self.relevance_threshold
                ]
                relevant_docs.extend(relevant)

            eval_dataset[query] = relevant_docs

        # Save evaluation dataset
        with open(self.results_dir / "evaluation_dataset.json", "w") as f:
            json.dump(eval_dataset, f, indent=2)

        return eval_dataset

    def calculate_metrics(
        self, method: str, query_results: Dict[str, List[Dict]]
    ) -> Dict[str, float]:
        """
        Calculate evaluation metrics for a similarity method.

        Args:
            method: Name of similarity method
            query_results: Dictionary mapping queries to ranked results

        Returns:
            Dictionary of metric names to values
        """
        metrics = {
            "precision@k": [],
            "recall@k": [],
            "diversity_score": [],
            "mean_reciprocal_rank": [],
        }

        for query, results in query_results.items():
            # Get ground truth
            relevant_docs = set(
                d["doc_id"] for d in self.metrics_cache["eval_dataset"][query]
            )

            # Calculate metrics
            metrics["precision@k"].append(
                calculate_precision_at_k(
                    [r["doc_id"] for r in results[: self.top_k]], relevant_docs
                )
            )

            metrics["recall@k"].append(
                calculate_recall_at_k(
                    [r["doc_id"] for r in results[: self.top_k]], relevant_docs
                )
            )

            metrics["diversity_score"].append(
                calculate_diversity_score(results[: self.top_k])
            )

            # Calculate MRR
            for rank, result in enumerate(results, 1):
                if result["doc_id"] in relevant_docs:
                    metrics["mean_reciprocal_rank"].append(1.0 / rank)
                    break
            else:
                metrics["mean_reciprocal_rank"].append(0.0)

        # Average metrics across queries
        return {metric: np.mean(values) for metric, values in metrics.items()}

    def run_comparison(self) -> pd.DataFrame:
        """
        Run comparison of all similarity methods.

        Returns:
            DataFrame with comparison results
        """
        logger.info("Running similarity methods comparison...")

        # Build or load evaluation dataset
        if "eval_dataset" not in self.metrics_cache:
            self.metrics_cache["eval_dataset"] = self.build_evaluation_dataset()

        results = []
        for method, calculator in self.similarity_calculators.items():
            logger.info(f"Evaluating {method} similarity...")

            # Get search results for all queries
            query_results = {}
            for query in tqdm(self.evaluation_queries, desc=f"Processing {method}"):
                results_list = self.document_retriever.retrieve_documents(
                    query, calculator, top_k=self.top_k
                )
                query_results[query] = results_list

            # Calculate metrics
            metrics = self.calculate_metrics(method, query_results)
            metrics["method"] = method
            results.append(metrics)

        # Create comparison DataFrame
        df = pd.DataFrame(results)

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_csv(self.results_dir / f"comparison_results_{timestamp}.csv", index=False)

        return df

    def generate_report(self, comparison_df: pd.DataFrame) -> str:
        """
        Generate evaluation report with insights and visualizations.

        Args:
            comparison_df: DataFrame with comparison results

        Returns:
            Path to generated report
        """
        logger.info("Generating evaluation report...")

        report_dir = self.results_dir / "reports"
        report_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = report_dir / f"evaluation_report_{timestamp}.html"

        # Create visualizations
        self._create_metrics_comparison_plot(comparison_df)
        self._create_method_ranking_plot(comparison_df)

        # Generate HTML report
        html_content = self._generate_html_report(comparison_df)

        with open(report_path, "w") as f:
            f.write(html_content)

        return str(report_path)

    def run_ab_test(
        self, method_a: str, method_b: str, metric: str = "precision@k"
    ) -> Dict[str, Union[float, bool]]:
        """
        Run statistical A/B test between two methods.

        Args:
            method_a: First method to compare
            method_b: Second method to compare
            metric: Metric to compare

        Returns:
            Dictionary with test results
        """
        logger.info(f"Running A/B test: {method_a} vs {method_b} on {metric}")

        # Get metric values for both methods
        values_a = self._get_metric_values(method_a, metric)
        values_b = self._get_metric_values(method_b, metric)

        # Run t-test
        t_stat, p_value = stats.ttest_ind(values_a, values_b)

        return {
            "t_statistic": t_stat,
            "p_value": p_value,
            "significant": p_value < 0.05,
            "better_method": method_a if t_stat > 0 else method_b,
        }

    def _calculate_basic_relevance(self, query: str, doc_text: str) -> float:
        """
        Calculate basic relevance score between query and document.
        This is a simple implementation - in production, use human annotations.
        """
        query_terms = set(query.lower().split())
        doc_terms = set(doc_text.lower().split())

        overlap = len(query_terms & doc_terms)
        return overlap / len(query_terms)

    def _get_metric_values(self, method: str, metric: str) -> List[float]:
        """Get all values for a specific metric and method."""
        if "metric_values" not in self.metrics_cache:
            self.metrics_cache["metric_values"] = {}

        cache_key = f"{method}_{metric}"
        if cache_key not in self.metrics_cache["metric_values"]:
            values = []
            for query in self.evaluation_queries:
                results = self.document_retriever.retrieve_documents(
                    query, self.similarity_calculators[method], top_k=self.top_k
                )
                relevant_docs = set(
                    d["doc_id"] for d in self.metrics_cache["eval_dataset"][query]
                )

                if metric == "precision@k":
                    value = calculate_precision_at_k(
                        [r["doc_id"] for r in results[: self.top_k]], relevant_docs
                    )
                elif metric == "recall@k":
                    value = calculate_recall_at_k(
                        [r["doc_id"] for r in results[: self.top_k]], relevant_docs
                    )
                elif metric == "diversity_score":
                    value = calculate_diversity_score(results[: self.top_k])

                values.append(value)

            self.metrics_cache["metric_values"][cache_key] = values

        return self.metrics_cache["metric_values"][cache_key]

    def _create_metrics_comparison_plot(self, df: pd.DataFrame):
        """Create comparison plot for all metrics."""
        metrics = [col for col in df.columns if col != "method"]

        plt.figure(figsize=(12, 6))
        x = np.arange(len(df["method"]))
        width = 0.8 / len(metrics)

        for i, metric in enumerate(metrics):
            plt.bar(x + i * width, df[metric], width, label=metric)

        plt.xlabel("Similarity Method")
        plt.ylabel("Score")
        plt.title("Comparison of Similarity Methods")
        plt.xticks(x + width * (len(metrics) - 1) / 2, df["method"], rotation=45)
        plt.legend()
        plt.tight_layout()

        plt.savefig(self.results_dir / "metrics_comparison.png")
        plt.close()

    def _create_method_ranking_plot(self, df: pd.DataFrame):
        """Create ranking plot for methods."""
        metrics = [col for col in df.columns if col != "method"]
        rankings = df[metrics].rank(ascending=False)
        rankings["method"] = df["method"]

        plt.figure(figsize=(10, 6))
        sns.heatmap(
            rankings.set_index("method")[metrics], annot=True, cmap="YlOrRd", fmt=".0f"
        )
        plt.title("Method Rankings by Metric")
        plt.tight_layout()

        plt.savefig(self.results_dir / "method_rankings.png")
        plt.close()

    def _generate_html_report(self, df: pd.DataFrame) -> str:
        """Generate HTML report content."""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Legal Document Search Evaluation Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                img { max-width: 100%; margin: 20px 0; }
                .section { margin: 20px 0; }
            </style>
        </head>
        <body>
            <h1>Legal Document Search Evaluation Report</h1>
            <div class="section">
                <h2>Summary</h2>
                <p>Evaluation performed on {num_queries} queries across {num_methods} similarity methods.</p>
                {summary_table}
            </div>

            <div class="section">
                <h2>Visualizations</h2>
                <h3>Metrics Comparison</h3>
                <img src="metrics_comparison.png" alt="Metrics Comparison">

                <h3>Method Rankings</h3>
                <img src="method_rankings.png" alt="Method Rankings">
            </div>

            <div class="section">
                <h2>Recommendations</h2>
                <p>{recommendations}</p>
            </div>

            <div class="section">
                <h2>A/B Test Results</h2>
                {ab_test_results}
            </div>

            <footer>
                <p>Generated on: {timestamp}</p>
            </footer>
        </body>
        </html>
        """

        # Generate summary table
        summary_table = df.to_html(index=False)

        # Generate recommendations
        best_method = df.loc[df["precision@k"].idxmax(), "method"]
        recommendations = f"""
        Based on the evaluation results:
        - The {best_method} method shows the best overall performance
        - Key strengths and weaknesses of each method:
        """
        for method in df["method"]:
            metrics = df[df["method"] == method].iloc[0]
            recommendations += f"\n  - {method}: "
            recommendations += "Strong in " + ", ".join(
                m for m in metrics.index if m != "method" and metrics[m] >= df[m].mean()
            )

        # Generate A/B test results
        ab_results = []
        methods = df["method"].tolist()
        for i, method_a in enumerate(methods):
            for method_b in methods[i + 1 :]:
                result = self.run_ab_test(method_a, method_b)
                ab_results.append(
                    f"<p><strong>{method_a} vs {method_b}</strong>:<br>"
                    f"Better method: {result['better_method']}<br>"
                    f"Statistically significant: {'Yes' if result['significant'] else 'No'}"
                    f" (p={result['p_value']:.3f})</p>"
                )

        return html_template.format(
            num_queries=len(self.evaluation_queries),
            num_methods=len(df),
            summary_table=summary_table,
            recommendations=recommendations,
            ab_test_results="".join(ab_results),
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )


def main():
    """Run evaluation pipeline with example usage."""
    # Initialize required components
    data_loader = DataLoader(DATA_DIR)
    document_retriever = DocumentRetriever()

    # Create and run evaluation pipeline
    pipeline = EvaluationPipeline(data_loader, document_retriever)

    # Run comparison
    comparison_results = pipeline.run_comparison()

    # Generate report
    report_path = pipeline.generate_report(comparison_results)

    # Run A/B test example
    ab_test_result = pipeline.run_ab_test("cosine", "hybrid")

    logger.info(f"Evaluation completed. Report generated at: {report_path}")
    logger.info(f"A/B test result: {ab_test_result}")


if __name__ == "__main__":
    main()
