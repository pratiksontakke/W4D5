"""
Integration tests for the Legal Document Search System.

Tests end-to-end functionality and component interactions.
"""
import logging
import os
from typing import Dict, List

import pytest
from config import EVALUATION_CONFIG
from core.evaluation import evaluate_search_results
from fastapi.testclient import TestClient
from pipelines.build_index import build_search_index
from pipelines.evaluate import run_evaluation
from tests.test_config import TEST_DATA_DIR
from ui.web_app import app

logger = logging.getLogger(__name__)

# ... rest of the file ...
