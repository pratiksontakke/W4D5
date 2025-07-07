"""
Tests for the evaluation pipeline.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.data_loader import DataLoader
from core.retrieval import DocumentRetriever
from pipelines.evaluate import EvaluationPipeline

# Test data
TEST_QUERIES = ["Income tax deduction for education", "GST rate for textile products"]


@pytest.fixture
def evaluation_pipeline():
    """Create an evaluation pipeline instance for testing."""
    data_loader = DataLoader(Path("data"))
    document_retriever = DocumentRetriever()
    return EvaluationPipeline(
        data_loader=data_loader,
        document_retriever=document_retriever,
        evaluation_queries=TEST_QUERIES,
    )


def test_build_evaluation_dataset(evaluation_pipeline, tmp_path):
    """Test building evaluation dataset."""
    # Set temporary results directory
    evaluation_pipeline.results_dir = tmp_path

    # Build dataset
    eval_dataset = evaluation_pipeline.build_evaluation_dataset()

    # Check dataset structure
    assert isinstance(eval_dataset, dict)
    assert all(query in eval_dataset for query in TEST_QUERIES)
    assert all(isinstance(docs, list) for docs in eval_dataset.values())

    # Check saved file
    dataset_file = tmp_path / "evaluation_dataset.json"
    assert dataset_file.exists()

    with open(dataset_file) as f:
        saved_dataset = json.load(f)
    assert saved_dataset == eval_dataset


def test_calculate_metrics(evaluation_pipeline):
    """Test metrics calculation."""
    # Mock query results
    query_results = {
        TEST_QUERIES[0]: [
            {"doc_id": "doc1", "score": 0.9},
            {"doc_id": "doc2", "score": 0.8},
        ]
    }

    # Mock evaluation dataset
    evaluation_pipeline.metrics_cache["eval_dataset"] = {
        TEST_QUERIES[0]: [{"doc_id": "doc1", "relevance_score": 1.0}]
    }

    # Calculate metrics
    metrics = evaluation_pipeline.calculate_metrics("cosine", query_results)

    # Check metrics
    assert isinstance(metrics, dict)
    assert all(
        metric in metrics
        for metric in [
            "precision@k",
            "recall@k",
            "diversity_score",
            "mean_reciprocal_rank",
        ]
    )
    assert all(isinstance(score, float) for score in metrics.values())
    assert 0 <= metrics["precision@k"] <= 1
    assert 0 <= metrics["recall@k"] <= 1


def test_run_comparison(evaluation_pipeline, tmp_path):
    """Test running comparison of methods."""
    # Set temporary results directory
    evaluation_pipeline.results_dir = tmp_path

    # Run comparison
    df = evaluation_pipeline.run_comparison()

    # Check DataFrame structure
    assert isinstance(df, pd.DataFrame)
    assert "method" in df.columns
    assert all(
        method in df["method"].values
        for method in ["cosine", "euclidean", "mmr", "hybrid"]
    )

    # Check metrics columns
    metric_columns = [
        "precision@k",
        "recall@k",
        "diversity_score",
        "mean_reciprocal_rank",
    ]
    assert all(col in df.columns for col in metric_columns)

    # Check saved file
    csv_files = list(tmp_path.glob("comparison_results_*.csv"))
    assert len(csv_files) == 1
    saved_df = pd.read_csv(csv_files[0])
    pd.testing.assert_frame_equal(df, saved_df)


def test_generate_report(evaluation_pipeline, tmp_path):
    """Test report generation."""
    # Set temporary results directory
    evaluation_pipeline.results_dir = tmp_path
    report_dir = tmp_path / "reports"
    report_dir.mkdir()

    # Create test comparison results
    comparison_df = pd.DataFrame(
        {
            "method": ["cosine", "euclidean"],
            "precision@k": [0.8, 0.7],
            "recall@k": [0.75, 0.65],
            "diversity_score": [0.6, 0.7],
            "mean_reciprocal_rank": [0.85, 0.75],
        }
    )

    # Generate report
    report_path = evaluation_pipeline.generate_report(comparison_df)

    # Check report file
    assert Path(report_path).exists()
    assert report_path.endswith(".html")

    # Check visualization files
    assert (tmp_path / "metrics_comparison.png").exists()
    assert (tmp_path / "method_rankings.png").exists()


def test_run_ab_test(evaluation_pipeline):
    """Test A/B testing."""
    # Mock metric values
    evaluation_pipeline.metrics_cache["metric_values"] = {
        "cosine_precision@k": [0.8, 0.85, 0.75],
        "hybrid_precision@k": [0.9, 0.85, 0.95],
    }

    # Run A/B test
    result = evaluation_pipeline.run_ab_test("cosine", "hybrid")

    # Check result structure
    assert isinstance(result, dict)
    assert all(
        key in result
        for key in ["t_statistic", "p_value", "significant", "better_method"]
    )
    assert isinstance(result["significant"], bool)
    assert result["better_method"] in ["cosine", "hybrid"]


def test_basic_relevance_calculation(evaluation_pipeline):
    """Test basic relevance calculation."""
    query = "tax deduction"
    doc_text = "This document explains income tax deductions."

    score = evaluation_pipeline._calculate_basic_relevance(query, doc_text)

    assert isinstance(score, float)
    assert 0 <= score <= 1
    assert score > 0  # Should find some relevance
