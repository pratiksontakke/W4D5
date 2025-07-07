"""
Data processing and management components.
"""

from .dataset import SalesDataset
from .ingestion.transcript import TranscriptProcessor
from .ingestion.metadata import MetadataIntegrator
from .processing.cleaner import TextCleaner
from .processing.labeler import ConversionLabeler

__all__ = [
    'SalesDataset',
    'TranscriptProcessor',
    'MetadataIntegrator',
    'TextCleaner',
    'ConversionLabeler'
] 