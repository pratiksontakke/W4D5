import nltk

def download_nltk_data():
    """Download required NLTK data packages."""
    packages = ['punkt', 'stopwords']
    for package in packages:
        try:
            nltk.download(package, quiet=True)
            print(f"Successfully downloaded {package}")
        except Exception as e:
            print(f"Error downloading {package}: {e}")

if __name__ == "__main__":
    download_nltk_data() 