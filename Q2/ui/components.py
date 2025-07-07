"""
Reusable UI components for the Indian Legal Document Search System.
"""

from typing import Dict, List, Optional

from fastapi import Request
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel


class DocumentCard(BaseModel):
    """Model for document card data."""

    id: str
    title: str
    excerpt: str
    score: float
    document_type: str
    url: Optional[str] = None


class MetricsData(BaseModel):
    """Model for metrics data."""

    precision: float
    recall: float
    diversity: float
    response_time: float


class UIComponents:
    def __init__(self, templates: Jinja2Templates):
        self.templates = templates

    async def render_document_card(
        self, request: Request, document: DocumentCard
    ) -> str:
        """Render a document card component."""
        return self.templates.get_template("components/document_card.html").render(
            request=request, document=document
        )

    async def render_metrics_dashboard(
        self, request: Request, metrics: Dict[str, MetricsData]
    ) -> str:
        """Render the metrics dashboard component."""
        return self.templates.get_template("components/metrics_dashboard.html").render(
            request=request, metrics=metrics
        )

    async def render_upload_area(
        self, request: Request, allowed_types: List[str], max_size_mb: int
    ) -> str:
        """Render the file upload area component."""
        return self.templates.get_template("components/upload_area.html").render(
            request=request, allowed_types=allowed_types, max_size_mb=max_size_mb
        )

    async def render_search_form(
        self, request: Request, document_types: List[str]
    ) -> str:
        """Render the search form component."""
        return self.templates.get_template("components/search_form.html").render(
            request=request, document_types=document_types
        )

    async def render_results_grid(
        self, request: Request, results: Dict[str, List[DocumentCard]]
    ) -> str:
        """Render the results grid component."""
        return self.templates.get_template("components/results_grid.html").render(
            request=request, results=results
        )

    async def render_feedback_form(
        self, request: Request, document_id: str, query_id: str
    ) -> str:
        """Render the feedback form component."""
        return self.templates.get_template("components/feedback_form.html").render(
            request=request, document_id=document_id, query_id=query_id
        )

    @staticmethod
    def format_score(score: float) -> str:
        """Format similarity score for display."""
        return f"{score:.2f}"

    @staticmethod
    def get_score_class(score: float) -> str:
        """Get CSS class based on score value."""
        if score >= 0.8:
            return "score-high"
        elif score >= 0.5:
            return "score-medium"
        return "score-low"

    @staticmethod
    def truncate_text(text: str, max_length: int = 200) -> str:
        """Truncate text with ellipsis if too long."""
        if len(text) <= max_length:
            return text
        return text[: max_length - 3] + "..."
