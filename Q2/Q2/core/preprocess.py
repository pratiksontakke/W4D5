"""
Text preprocessing and normalization module.
"""
import logging
from typing import Dict, List

import spacy
from spacy.language import Language

from config import PROCESSING_CONFIG

logger = logging.getLogger(__name__)


def clean_legal_text(text: str, max_length: int = 1000) -> str:
    """
    Clean and normalize legal text while preserving important legal terminology.

    Args:
        text: Input text to clean
        max_length: Maximum length for text chunks
    """
    # Split into smaller chunks for processing
    chunks = [text[i : i + max_length] for i in range(0, len(text), max_length)]

    # Process each chunk
    processed_chunks = []
    for chunk in chunks:
        processed = chunk.strip()
        processed_chunks.append(processed)

    return " ".join(processed_chunks)


# ... rest of the file ...
