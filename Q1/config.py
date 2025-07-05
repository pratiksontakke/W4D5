import os
from pathlib import Path

# Project root directory
ROOT_DIR = Path(__file__).parent

# Directory for saving models
MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Directory for saving plots
PLOTS_DIR = ROOT_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# Categories for classification
CATEGORIES = ['Tech', 'Finance', 'Healthcare', 'Sports', 'Politics', 'Entertainment']

# OpenAI settings
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Training settings
SAMPLES_PER_CATEGORY = 25
RANDOM_SEED = 42 