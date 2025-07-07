"""
Setup script for the Indian Legal Document Search System.
"""

import subprocess
import sys


def main():
    """Install project dependencies and download required models."""
    print("Installing project dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    print("\nDownloading spaCy model...")
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_lg"])
    
    print("\nSetup completed successfully!")

if __name__ == "__main__":
    main()
