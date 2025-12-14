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
