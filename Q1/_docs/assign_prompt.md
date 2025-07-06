# The Universal AI Project SOP (Standard Operating Procedure)

This document contains a standardized chain of prompts designed to transform **any AI assignment** into a production-ready, well-documented, and maintainable project.

## The Philosophy
We move from the abstract "what" to the concrete "how" in a structured, repeatable sequence. Each step's output is a required input for the next, ensuring a robust and logical workflow. This is how you build professional-grade AI systems, not just one-off scripts.

---

### **Prompt 1: Project Deconstruction (The "What")**

**Objective:** To perform a deep analysis of the assignment, establishing a shared understanding of the project's core purpose, problem domain, and key conceptual pieces before any code is considered.

**When to Use:** Immediately after receiving a new assignment.

**Your Input (The prompt you will use):**

```
Role: You are a Principal AI Engineer. Your task is to deconstruct a new project assignment to ensure it is fully understood before development begins.

Assignment Details:
--- START OF ASSIGNMENT ---
[PASTE YOUR ENTIRE ASSIGNMENT TEXT HERE]
--- END OF ASSIGNMENT ---

Based on the assignment text, generate the following Deconstruction Report in clean Markdown:

1.  **Core Task:** In one simple sentence, what is the fundamental goal?
2.  **Problem Type:** What is the formal name for this kind of problem? (e.g., Semantic Search, Text Classification, Time-Series Forecasting, Recommender System, Object Detection).
3.  **Key Components (Mind Map):** Identify all the essential conceptual parts needed to solve this. List them abstractly. (e.g., Input Data Source, Text Representation Method, Core Algorithm, Evaluation Metric, User Interface).
4.  **The "Motto" (Real-World Value):** Explain why this task is valuable. Provide 3 concrete, real-world examples where this exact technology would be used. Create a single, motivating "motto" for the project.
```

**Your Action:** Save the output as `_docs/1_DECONSTRUCTION.md`.

---

### **Prompt 2: Architectural Strategy (The "Where")**

**Objective:** To map the abstract components from the Deconstruction Report onto a standardized, production-ready file structure. This defines *where* each piece of logic will live.

**When to Use:** Immediately after completing Prompt 1.

**Your Input (The prompt you will use):**

```
Role: You are a Lead AI Architect. I have a Deconstruction Report for a new project. Your task is to create a strategic architectural plan.

Here is the Deconstruction Report:
--- START OF REPORT ---
[PASTE THE ENTIRE OUTPUT FROM PROMPT 1 HERE]
--- END OF REPORT ---

Based on this report, generate the following Architectural Strategy document in Markdown:

1.  **Proposed Architecture:** Present a standardized, layered blueprint for this project using a directory structure (e.g., `core/`, `pipelines/`, `ui/`, `config.py`).
2.  **Component Mapping:** For each "Key Component" identified in the report, explicitly assign it a home within the proposed architecture. (e.g., "The 'Text Representation Method' will be handled in `core/embedders.py`," "The 'Core Algorithm' will be implemented in `core/similarity.py`," "The 'User Interface' will be built in `ui/web_app.py`").
3.  **Data & Logic Flow:** Describe the project's two main operational phases:
    *   **Offline Phase (Preparation/Training):** Explain the sequence of events that a `pipelines/` script will execute to process data and save necessary artifacts.
    *   **Online Phase (Inference/Serving):** Explain how the main application (`app.py` or `ui/web_app.py`) will load the saved artifacts and use them to perform its core task on new input.
```

**Your Action:** Save the output as `_docs/2_STRATEGY.md`.

---

### **Prompt 3: Dynamic Project Planning (The "How-To")**

**Objective:** To generate dynamic, project-specific planning documents. These are derived directly from the project's unique components, not from a fixed template.

**When to Use:** After completing Prompt 2.

**Your Input (The prompt you will use):**

```
Role: You are a Senior AI Project Manager and a Principal Scientist. Your task is to create detailed planning and educational documents based on the project's confirmed architecture.

Here are the project planning documents:
--- START OF DECONSTRUCTION REPORT ---
[PASTE THE ENTIRE OUTPUT FROM PROMPT 1 HERE]
--- END OF DECONSTRUCTION REPORT ---

--- START OF ARCHITECTURAL STRATEGY ---
[PASTE THE ENTIRE OUTPUT FROM PROMPT 2 HERE]
--- END OF ARCHITECTURAL STRATEGY ---

Generate the following two documents in a single response, separated by a clear heading.

**DOCUMENT 1: Project Plan & Feature Checklist**
Create a granular, phase-based project plan as a Markdown checklist. The plan must be derived directly from the `Component Mapping` and `Data & Logic Flow` sections of the Architectural Strategy.
- Create a Phase for each major part of the architecture (e.g., Setup, Core Logic, Pipelines, UI, Evaluation).
- For the `Core Logic` phase, create one checklist item for *each file* mentioned in the `Component Mapping`.

**DOCUMENT 2: Technical Deep-Dive**
For each abstract `Key Component` from the Deconstruction Report that represents a core algorithm, model, or complex technical method, create a detailed explanation with the following structure:
1.  **Component Name:** (e.g., `### Cosine Similarity`)
2.  **Core Principle:** What is the high-level concept?
3.  **Specific Logic:** How does it work in simple terms?
4.  **Common Tooling:** What libraries are typically used?
5.  **Project Motto:** A single sentence on the key learning objective for this specific component.
```

**Your Action:**
1.  Save "Document 1" as `_docs/3_FEATURE_CHECKLIST.md`.
2.  Save "Document 2" as `_docs/4_TECHNICAL_DEEP_DIVE.md`.

---

### **Prompt 4+: Component Implementation (The "Code")**

**Objective:** To generate the actual, production-quality code for each file defined in the architecture, one file at a time.

**When to Use:** Sequentially, for each file you need to build.

**Your Input (The prompt you will use):**

```
Role: You are a Senior AI Engineer. Your task is to implement a specific component of our project according to our established architecture and configuration.

Here are our project's foundational documents:
--- ARCHITECTURAL STRATEGY (SUMMARY) ---
[PASTE THE 'Component Mapping' AND 'Data & Logic Flow' SECTIONS FROM PROMPT 2's OUTPUT]
--- CONFIGURATION FILE (`config.py`) ---
[PASTE THE FULL CONTENT OF YOUR `config.py` FILE ONCE IT'S CREATED]

**Your Task:**
Write the complete, production-ready Python code for the following file: **[Specify the file path, e.g., `core/data_loader.py`]**.

**Code Requirements:**
1.  It must be fully functional and self-contained for its defined purpose.
2.  It must import any necessary settings from the provided `config.py` (if applicable).
3.  It must be well-commented, explaining the "why" behind key sections of code.
4.  It must adhere to the purpose defined for it in the Architectural Strategy.
5.  Include a `if __name__ == '__main__':` block with a simple example or test case to demonstrate its functionality.
```

**Your Action:** Create the specified file and paste the generated code into it. Review and understand the code. Repeat for all necessary files.

---

### **The Final Prompt: README Generation (The "Story")**

**Objective:** To generate a comprehensive, user-facing README file that tells the story of your project.

**When to Use:** When all code is complete and working.

**Your Input (The prompt you will use):**

```
Role: You are a Technical Writer specializing in AI/ML projects. I have a completed project and need a high-quality README file.

Project Context:
- Core Task: [Provide the one-sentence summary from Prompt 1]
- Key Technical Components: [List the key models/algorithms from the Deconstruction Report]
- Project Structure: [Briefly describe the `core/`, `pipelines/` structure]

Generate a comprehensive `README.md` file that includes:
1.  A clear project title and a short, engaging overview.
2.  A "Features" section listing the key capabilities.
3.  A "Project Architecture" section briefly explaining the blueprint.
4.  A "How It Works" section explaining the offline and online phases.
5.  A "Setup and Installation" section with clear, numbered steps.
6.  A "Usage" section explaining how to run the main pipeline(s) and the final application.
7.  A "Results and Analysis" section (you can add a placeholder for me to fill in).
```

**Your Action:** Create/overwrite your root `README.md` file with the generated content and fill in any final details. 