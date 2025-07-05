# Blueprint Step 3: Execution (The 'Do')

This is the final stage where you write the code. Because you have a clear plan from the previous stages, this becomes a methodical process of filling in the blanks, not a chaotic exploration.

## The Standard Operating Procedure (SOP)

Follow these steps in order for a clean and predictable build.

### Step 0: The Grand Plan - Setup the Blueprint
1.  **Action:** Create the full, empty directory structure (`_docs/`, `core/`, `pipelines/`, etc.).
2.  **Action:** Initialize `git` and create your `requirements.txt`.
3.  **Output:** A clean project skeleton.

### Step 1: The Foundation - Configure Everything
1.  **Action:** Open `config.py`. Define all paths, model names, API keys, and other parameters.
2.  **Output:** A central "control panel" for your entire project.

### Step 2: The Logic - Build the Core Engine
1.  **Action:** Go to `core/embedders.py`. Define the base `Embedder` class. Implement a concrete class for each embedding model (e.g., `OpenAIEmbedder`, `SentenceBertEmbedder`).
2.  **Action:** Go to `core/classifier.py`. Implement the `UnifiedClassifier` that takes an `Embedder` object.
3.  **Output:** A library of powerful, interchangeable, and well-contained components.

### Step 3: The Process - Write the Pipeline
1.  **Action:** Go to `pipelines/1_train_and_evaluate.py`.
2.  **Action:** Import components from `core/` and settings from `config.py`. Write the main loop that instantiates, trains, evaluates, and saves each model.
3.  **Output:** Trained model artifacts (`.joblib` files) in the `/models` directory.

### Step 4: The Showcase - Build the UI
1.  **Action:** Go to `app.py`.
2.  **Action:** Write the Streamlit UI. The logic should simply load the saved artifacts from the `/models` directory and call their `.predict()` method.
3.  **Output:** A working, interactive web application.

---
**Output of this Stage:** A complete, working, and maintainable application that perfectly matches the architecture defined in the strategy phase. 