# Your Personal "Prompt Engineering SOP for Applied AI"

This is a fantastic and highly professional way to think. You're asking to build a meta-skill: the ability to use a Prompt Engineer's mindset to systematically solve any given assignment. You want a personal "Prompt SOP" (Standard Operating Procedure).

This is exactly how you take back control and ensure every project is built on a solid, repeatable foundation.

Here is your Standardized Prompt Chain for Production-Ready AI Assignments. When you get a new assignment, you will start with Prompt 1 and proceed sequentially.

## The Goal
To transform any assignment into a production-ready, well-documented solution using a structured, prompt-driven workflow.

## Prompt 1: Project Deconstruction & Mind Mapping

**Objective:** To force a deep, initial analysis of the assignment before thinking about code, establishing a clear "what" and "why".

**When to Use:** Immediately after receiving a new assignment.

**Your Input (The prompt you will use):**

```
Role: You are a Principal AI Engineer. Your task is to deconstruct a new project assignment for a junior engineer to ensure they understand it deeply before starting.

Assignment Details:
I have a new assignment. Please analyze it and provide a structured "Deconstruction Report".

--- START OF ASSIGNMENT ---
[PASTE YOUR ENTIRE ASSIGNMENT TEXT HERE]
--- END OF ASSIGNMENT ---

Based on the assignment text, generate the following Deconstruction Report:

1. **Core Task:** In one simple sentence, what is the fundamental goal?
2. **Problem Type:** What is the formal name for this kind of problem (e.g., Text Classification, Object Detection, Time-Series Forecasting)?
3. **Key Components (Mind Map):** Identify the essential conceptual parts needed to solve this. List them out (e.g., Input Data, Transformation Method, Learning Algorithm, etc.).
4. **The "Motto" (Real-World Value):** Explain why this task is valuable. Provide 3 concrete, real-world examples where this exact technology would be used in a production environment. Create a single, motivating "motto" for the project.

The output should be in clean, well-formatted Markdown.
```

**Expected AI Output:** A detailed report with the four sections filled out, giving you a clear, high-level understanding of the project.

**Your Action:**
1. Create a folder named `_docs/` in your new project repository
2. Create a file named `_docs/1_DECONSTRUCTION.md`
3. Paste the AI's output into this file

**Pro Tip:** This first step is the most important. It prevents you from rushing into code and forces you to build a strong mental model of the project's purpose.

## Prompt 2: Architectural Strategy & Blueprint Mapping

**Objective:** To translate the "what" from Prompt 1 into a concrete "how" by mapping it onto a standardized, production-ready architecture.

**When to Use:** After you have the Deconstruction Report from Prompt 1.

**Your Input (The prompt you will use):**

```
Role: You are a Lead AI Architect. I have a Deconstruction Report for a new project. Your task is to create a strategic architectural plan.

Here is the Deconstruction Report:
--- START OF REPORT ---
[PASTE THE ENTIRE OUTPUT FROM PROMPT 1 HERE]
--- END OF REPORT ---

Based on this report, generate the following Architectural Strategy document in Markdown:

1. **Proposed Architecture:** Present the standardized, layered blueprint for this project. Show the directory structure (`core/`, `pipelines/`, `config.py`, etc.).
2. **Component Mapping:** For each "Key Component" identified in the report, explicitly assign it a home within the proposed architecture. (e.g., "The 'OpenAI Embedding' component will be implemented as the `OpenAIEmbedder` class inside `core/embedders.py`.").
3. **Data & Logic Flow:** Describe the two main phases of the project:
   * **Offline Phase (Training):** Explain the sequence of events that the `pipelines/` script will execute to train and save the model artifacts.
   * **Online Phase (Inference):** Explain how the `app.py` will load the saved artifacts and use them to make predictions.

This document will serve as the official blueprint for building the project.
```

**Expected AI Output:** A clear architectural document showing the file structure and explaining where each part of the project's logic will live.

**Your Action:**
1. Create a file named `_docs/2_STRATEGY.md`
2. Paste the AI's output into this file

**Pro Tip:** This step solidifies the project structure before implementation, preventing disorganized code and making the project instantly more maintainable.

## Prompt 3: Scaffolding & Configuration

**Objective:** To generate the initial project skeleton and the central configuration file.

**When to Use:** After you have the Architectural Strategy from Prompt 2.

**Your Input (The prompt you will use):**

```
Role: You are a Senior DevOps Engineer specializing in MLOps. Your task is to set up the initial project structure based on the architectural plan.

Here is the Architectural Plan:
--- START OF PLAN ---
[PASTE THE ENTIRE OUTPUT FROM PROMPT 2 HERE]
--- END OF PLAN ---

And here are the key parameters from the original assignment:
- Models to implement: [List the models, e.g., Word2Vec, BERT, Sentence-BERT, OpenAI]
- Key model names/APIs: [List the specific names, e.g., 'all-MiniLM-L6-v2', 'text-embedding-ada-002']
- Categories for classification: [List the categories, e.g., Tech, Finance, Sports, Politics]

Based on all the above, generate the following:

1. **Project Skeleton:** Provide a shell command (`mkdir`, `touch`) to create the entire directory and file structure (including the `_docs/` folder).
2. **`config.py`:** Write the complete Python code for the `config.py` file. It should use `pathlib` for path management and contain all the configurable parameters (model names, category lists, file paths, etc.) identified from the assignment.
3. **`requirements.txt`:** List all the necessary Python libraries for this project.
```

**Expected AI Output:** The exact commands and code to create your project's foundation.

**Your Action:**
1. Run the shell commands in your terminal to create the project structure
2. Create config.py and requirements.txt and paste the generated code into them
3. Run pip install -r requirements.txt

**Pro Tip:** By creating config.py now, you are committing to a "no hardcoded values" policy from the very beginning.

## Prompt 4, 5, 6... (Implementation Prompts)

From here, you will generate the code for each component identified in your blueprint.

**Generic Implementation Prompt Template:**

```
Role: You are a Senior AI Engineer. Your task is to implement a specific component of our project according to our established architecture.

Architectural Plan:
[PASTE A SUMMARY OF THE ARCHITECTURAL PLAN FROM PROMPT 2]

Configuration File (`config.py`):
```python
[PASTE THE CONTENT OF YOUR config.py FILE]
```

Your Task:
Write the complete, production-ready Python code for the following file: [Specify the file path, e.g., core/embedders.py].

Requirements for the code:
1. It must be fully functional and self-contained
2. It must import its settings from the provided config.py
3. It must be well-commented, explaining the "why" behind the code
4. For core/ components, it must follow the defined class structure (e.g., inherit from a base Embedder class)
5. For pipelines/ components, it should clearly show the main execution loop
6. For app.py, it should load the models from the paths defined in config and create an intuitive UI
```

**Your Action:**
1. Use this template sequentially for each file you need to build: `core/embedders.py`, `core/classifier.py`, `pipelines/1_train_and_evaluate.py`, `app.py`, etc.
2. Create the file and paste the generated code into it. Review the code to ensure you understand it.

**Pro Tip:** By providing the `config.py` content in every implementation prompt, you ensure the AI consistently uses your central configuration, reinforcing the production-ready pattern.

## Prompt 7: Final Documentation

**Objective:** To generate the final user-facing documentation for your repository.

**When to Use:** When all the code is complete and working.

**Your Input (The prompt you will use):**

```
Role: You are a Technical Writer specializing in AI/ML projects. I have a completed project and need a high-quality README file.

Project Context:
- Core Task: [Provide the one-sentence summary from Prompt 1]
- Models Used: [List the models, e.g., Word2Vec, BERT, Sentence-BERT, OpenAI]
- Key Files: app.py, config.py, core/, pipelines/

Generate a comprehensive README.md file that includes:
1. A clear project title and a short, engaging overview
2. A "Features" section listing the key capabilities (e.g., "Compares 4 Embedding Models," "Real-time Classification UI")
3. A "Project Architecture" section briefly explaining the blueprint
4. A "Setup and Installation" section with clear, numbered steps
5. A "Usage" section explaining how to run the training pipeline and the web app
6. A "Results and Analysis" section (you can add a placeholder for this) summarizing which model performed best
```

**Your Action:**
1. Create/overwrite your `README.md` file with the generated content
2. Fill in any placeholder sections with your specific results

By following this prompt chain, you are not just completing an assignment. You are executing a professional workflow, building a maintainable product, and creating a repository of your own thinking. This is how you master the craft. 