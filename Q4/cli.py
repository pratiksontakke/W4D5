"""
Command-line interface for the Intelligent Document Chunking System.
"""
import argparse
import json
from pathlib import Path
import sys
import logging
from datetime import datetime

from core.processor import DocumentProcessor
from config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def process_document(args):
    """Process a single document."""
    processor = DocumentProcessor()
    file_path = Path(args.input)
    
    try:
        # Validate document
        if not processor.validate_document(file_path):
            logger.error(f"Invalid or unsupported document: {file_path}")
            return 1
            
        # Process document
        result = processor.process_document(file_path)
        
        # Save or display results
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
            logger.info(f"Results saved to {output_path}")
        else:
            # Print summary to console
            print("\nProcessing Results:")
            print(f"Document Type: {result['document_info']['type']}")
            print(f"Chunks Generated: {result['metadata']['chunk_count']}")
            print(f"Processing Time: {result['metadata']['processing_time']:.2f}s")
            print("\nClassification:")
            print(f"- Type: {result['classification']['document_type']}")
            print(f"- Confidence: {result['classification']['confidence']:.2%}")
            print(f"- Chunking Strategy: {result['classification']['chunking_strategy']['method']}")
            
        return 0
        
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        return 1

def process_directory(args):
    """Process all documents in a directory."""
    processor = DocumentProcessor()
    input_dir = Path(args.input)
    
    if not input_dir.is_dir():
        logger.error(f"Not a directory: {input_dir}")
        return 1
        
    # Prepare output directory
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Track statistics
    stats = {
        'total': 0,
        'successful': 0,
        'failed': 0,
        'start_time': datetime.now()
    }
    
    try:
        # Process each supported file
        for file_path in input_dir.rglob('*'):
            if file_path.suffix.lower() in processor.get_supported_types():
                stats['total'] += 1
                logger.info(f"Processing {file_path}")
                
                try:
                    result = processor.process_document(file_path)
                    stats['successful'] += 1
                    
                    # Save results if output directory specified
                    if args.output:
                        output_path = output_dir / f"{file_path.stem}_chunks.json"
                        with open(output_path, 'w') as f:
                            json.dump(result, f, indent=2)
                            
                except Exception as e:
                    logger.error(f"Failed to process {file_path}: {e}")
                    stats['failed'] += 1
                    
        # Calculate processing time
        processing_time = (datetime.now() - stats['start_time']).total_seconds()
        
        # Print summary
        print("\nProcessing Summary:")
        print(f"Total Files: {stats['total']}")
        print(f"Successful: {stats['successful']}")
        print(f"Failed: {stats['failed']}")
        print(f"Total Time: {processing_time:.2f}s")
        
        return 0 if stats['failed'] == 0 else 1
        
    except Exception as e:
        logger.error(f"Error processing directory: {e}")
        return 1

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Intelligent Document Chunking System"
    )
    
    parser.add_argument(
        'input',
        help='Input file or directory path'
    )
    
    parser.add_argument(
        '-o', '--output',
        help='Output file or directory path for results'
    )
    
    parser.add_argument(
        '-r', '--recursive',
        action='store_true',
        help='Process directory recursively'
    )
    
    args = parser.parse_args()
    
    # Determine processing mode
    input_path = Path(args.input)
    if input_path.is_file():
        return process_document(args)
    elif input_path.is_dir() and args.recursive:
        return process_directory(args)
    else:
        logger.error(f"Invalid input path or missing --recursive flag for directory: {input_path}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 