# 🩺 Medical Oncology RAG System with Reinforcement Learning

## 📌 Overview

This project is an AI-powered Medical Oncology Question Answering System built using Retrieval-Augmented Generation (RAG) and Reinforcement Learning.

The system retrieves relevant information from medical oncology textbooks and generates accurate answers using MedGemma running locally through Ollama.

---

## 🚀 Features

* Hybrid Retrieval (BM25 + Vector Search)
* ChromaDB Vector Database
* Medical Question Answering
* Reinforcement Learning Memory
* Self-Improving AI Responses
* FastAPI Backend
* Local LLM Inference using Ollama
* Automatic Evaluation Pipeline
* Semantic Search using Sentence Transformers

---

## 🛠️ Tech Stack

* Python
* FastAPI
* ChromaDB
* Sentence Transformers
* Ollama
* MedGemma
* Llama3 (Judge LLM)
* BM25 Retrieval
* NumPy
* PyTorch

---

## 📂 Project Pipeline

User Query → Query Analyzer Agent → Hybrid Retrieval (BM25 + Embeddings) → ChromaDB → Context Retrieval → MedGemma LLM → Reinforcement Learning Memory → Final Answer

---

## 📊 Evaluation Metrics

* Precision@5
* Recall@5
* MRR
* NDCG@5
* BLEU
* ROUGE
* METEOR
* BERTScore
* Faithfulness
* S.C.O.P.E Evaluation

---

## ▶️ Run the Project

```bash
pip install -r requirements.txt
python main.py
```

---

## 🌐 Run API

```bash
uvicorn api:app --reload
```

Open:

http://127.0.0.1:8000/docs

---

## 👨‍💻 Author

**Punith Kumar Reddy**
B.Tech Student | AI & Machine Learning Enthusiast
