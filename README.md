# LLM Evaluation Pipeline (RAG Quality Metrics)

A lightweight LLM evaluation pipeline that measures **relevance** and **hallucination** of chatbot responses using embedding-based semantic similarity.

This project simulates a real-world **RAG (Retrieval-Augmented Generation)** evaluation workflow with cost and latency tracking.

---

## Features

- Relevance scoring between user query, context, and LLM response
- Hallucination detection using semantic similarity
- Defensive handling for noisy / incomplete vector data
- Local execution (no paid API calls required)
- Tracks estimated latency and cost

---

## Tech Stack

- Python 3.10+
- SentenceTransformers (MiniLM)
- Transformers
- TensorFlow (CPU)
- JSON-based vector context

---

## Setup Instructions
1. Create virtual environment
> python -m venv venv

2. Activate virtual environment
   
- Windows
> venv\Scripts\activate

- macOS / Linux
> source venv/bin/activate

3. Install dependencies
> pip install -r requirements.txt

## Running the Pipeline

From the project root:
> python src/main.py


Output will be generated at:
> output/evaluation_results.json

## Evaluation Logic (High-Level)

- Relevance score is computed using cosine similarity between the user query and combined retrieved context.
- Hallucination score measures semantic distance between the LLM response and the supporting context chunks.
- Lower hallucination score indicates better grounding in retrieved data.

