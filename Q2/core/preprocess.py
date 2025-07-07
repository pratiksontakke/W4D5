"""
Module for text preprocessing and normalization.
"""
import logging
from typing import Dict, List

import spacy
from gensim.models import KeyedVectors
from spacy.language import Language

from config import PROCESSING_CONFIG

logger = logging.getLogger(__name__)

# Load spaCy model
nlp = spacy.load("en_core_web_lg")

# Load word vectors
word_vectors = KeyedVectors.load_word2vec_format(
    PROCESSING_CONFIG["word_vectors_path"], binary=True
)


def preprocess_text(text: str, max_length: int = 1000000) -> str:
    """
    Preprocess text by cleaning, normalizing and chunking.

    Args:
        text: Input text to preprocess
        max_length: Maximum length for spaCy processing

    Returns:
        Preprocessed text
    """
    # Split into smaller chunks for processing
    chunks = [text[i : i + max_length] for i in range(0, len(text), max_length)]

    # Process each chunk
    processed_chunks = []
    for chunk in chunks:
        doc = nlp(chunk)

        # Basic preprocessing
        tokens = [
            token.lemma_.lower()
            for token in doc
            if not token.is_stop and not token.is_punct and token.text.strip()
        ]

        processed_chunks.append(" ".join(tokens))

    return " ".join(processed_chunks)


def get_word_vector(word: str) -> List[float]:
    """
    Get word vector for a given word using word2vec model.

    Args:
        word: Input word

    Returns:
        Word vector as list of floats
    """
    try:
        return word_vectors[word].tolist()
    except KeyError:
        # Return zero vector if word not in vocabulary
        return [0.0] * word_vectors.vector_size
