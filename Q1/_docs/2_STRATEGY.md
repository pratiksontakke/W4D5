# Blueprint Step 2: Strategy (The 'How')

This stage maps the abstract components from the Deconstruction phase onto a concrete, standardized software architecture. This is where we decide *how* to build the system.

## The Standardized Architecture

This project follows a layered, production-ready blueprint to ensure it is modular, maintainable, and scalable.

```
/project/
├── app.py                  # Layer 4: User Interface
├── config.py               # Layer 5: Central Configuration
├── core/                   # Layer 1: Reusable Core Engine
│   ├── embedders.py
│   └── classifier.py
├── pipelines/              # Layer 2: Executable Recipes
│   └── 1_train_and_evaluate.py
└── models/                 # Layer 3: Saved Model Artifacts
```

## The Mental Checklist

### 1. How does each component fit the blueprint?
- **Action:** Take each node from your mind map (from Step 1) and assign it a home in the architecture.
- **Example Mapping:**
    - **Input/Data:** Belongs in a `core/data_loader.py`.
    - **Transformation/Embedding:** This is a core component. It gets its own class in `core/embedders.py` and must follow a standard interface.
    - **Learning Algorithm:** This is handled by the `core/classifier.py`, which is "injected" with an embedder.
    - **Training Process:** The orchestration of training belongs in a script in the `pipelines/` directory.
    - **User Interface:** This is the `app.py`.

### 2. What is the sequence of operations?
- **Action:** Define the distinct phases of the project lifecycle.
- **Example Sequence:**
    1.  **Offline Phase:** Run the `pipelines/` script. This script will use the `core/` components to train the model and save the final `.joblib` artifact into the `models/` directory.
    2.  **Online Phase:** Run the `app.py`. This application will load the pre-trained `.joblib` artifact from the `models/` directory and use it to make live predictions.

---
**Output of this Stage:** A complete architectural plan. You know which files to create and what the responsibility of each file is. You have a clear execution path. 