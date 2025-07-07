"""
FastAPI web application for the Indian Legal Document Search System.
Handles document upload, search queries, and results display.
"""

import logging
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import jwt
import uvicorn
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from config import LEGAL_DOCUMENT_TYPES, UI_CONFIG
from core.data_loader import DocumentLoader
from core.evaluation import MetricsCalculator
from core.similarity import SimilarityEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Indian Legal Document Search System",
    description="Compare different similarity methods for legal document retrieval",
    version="1.0.0",
)

# Security
security = HTTPBearer()
SECRET_KEY = "your-secret-key"  # In production, load from environment variable

# Setup static files and templates
app.mount("/static", StaticFiles(directory="ui/static"), name="static")
templates = Jinja2Templates(directory="ui/templates")

# Initialize core components
doc_loader = DocumentLoader()
similarity_engine = SimilarityEngine()
metrics_calculator = MetricsCalculator()

# Rate limiting
RATE_LIMIT = 100  # requests per minute
rate_limit_store: Dict[str, List[float]] = {}


def check_rate_limit(user_id: str) -> bool:
    """Check if user has exceeded rate limit."""
    current_time = time.time()
    if user_id not in rate_limit_store:
        rate_limit_store[user_id] = []

    # Remove old timestamps
    rate_limit_store[user_id] = [
        ts for ts in rate_limit_store[user_id] if current_time - ts < 60
    ]

    if len(rate_limit_store[user_id]) >= RATE_LIMIT:
        return False

    rate_limit_store[user_id].append(current_time)
    return True


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> dict:
    """Validate JWT token and return user info."""
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=["HS256"])
        if not check_rate_limit(payload["sub"]):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        return payload
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=401, detail="Invalid authentication credentials"
        )


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render home page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    document_type: str = Form(...),
    user: dict = Depends(get_current_user),
):
    """Handle document upload."""
    try:
        # Validate file size
        file_size = 0
        chunk_size = 1024 * 1024  # 1MB
        while chunk := await file.read(chunk_size):
            file_size += len(chunk)
            if file_size > UI_CONFIG["max_upload_size_mb"] * 1024 * 1024:
                raise HTTPException(status_code=413, detail="File too large")

        # Validate file extension
        if not any(
            file.filename.endswith(ext) for ext in UI_CONFIG["allowed_extensions"]
        ):
            raise HTTPException(status_code=400, detail="Invalid file type")

        # Save file
        file_path = Path(f"data/raw/{document_type}/{file.filename}")
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process document
        doc_loader.process_document(file_path)

        return {"message": "Document uploaded successfully"}

    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing document")


@app.post("/search")
async def search(
    query: str = Form(...),
    document_types: List[str] = Form(...),
    user: dict = Depends(get_current_user),
):
    """Handle search query and return results from all similarity methods."""
    try:
        # Validate document types
        if not all(dt in LEGAL_DOCUMENT_TYPES for dt in document_types):
            raise HTTPException(status_code=400, detail="Invalid document type")

        # Get results from each similarity method
        results = similarity_engine.search_all(
            query=query, document_types=document_types
        )

        # Calculate metrics
        metrics = metrics_calculator.calculate_metrics(query=query, results=results)

        return {"results": results, "metrics": metrics}

    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing search")


@app.post("/feedback")
async def submit_feedback(
    query_id: str = Form(...),
    relevant_docs: List[str] = Form(...),
    user: dict = Depends(get_current_user),
):
    """Handle user feedback on search results."""
    try:
        metrics_calculator.update_feedback(
            query_id=query_id, relevant_docs=relevant_docs, user_id=user["sub"]
        )
        return {"message": "Feedback recorded successfully"}

    except Exception as e:
        logger.error(f"Feedback error: {str(e)}")
        raise HTTPException(status_code=500, detail="Error recording feedback")


@app.get("/metrics")
async def get_metrics(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    user: dict = Depends(get_current_user),
):
    """Get performance metrics for the dashboard."""
    try:
        # Parse dates if provided
        start = datetime.fromisoformat(start_date) if start_date else None
        end = datetime.fromisoformat(end_date) if end_date else None

        metrics = metrics_calculator.get_performance_metrics(
            start_date=start, end_date=end
        )

        return metrics

    except Exception as e:
        logger.error(f"Metrics error: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving metrics")


if __name__ == "__main__":
    # Run the FastAPI application using uvicorn
    uvicorn.run("web_app:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
