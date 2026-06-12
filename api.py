from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime

from main import run, rl_memory, VALID_CATEGORIES

app = FastAPI(
    title="Medical Oncology RAG API",
    description="RAG + RL Medical Assistant",
    version="1.0"
)

class QuestionRequest(BaseModel):
    question: str


@app.get("/")
def home():
    return {
        "status": "running",
        "project": "Medical Oncology RAG + RL"
    }


@app.post("/ask")
def ask_question(request: QuestionRequest):

    answer = run(request.question)

    return {
        "question": request.question,
        "answer": answer,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/stats")
def get_stats():
    return rl_memory.get_stats()


@app.get("/categories")
def get_categories():
    return {
        "categories": VALID_CATEGORIES
    }


@app.post("/reset-memory")
def reset_memory():

    rl_memory.memory = []
    rl_memory.policy = {}
    rl_memory.save()

    return {
        "message": "RL memory cleared successfully"
    }


@app.get("/system-info")
def system_info():

    return {
        "llm": "medgemma",
        "vector_database": "chromadb",
        "embedding_model": "all-MiniLM-L6-v2",
        "retrieval": "hybrid_search",
        "reinforcement_learning": True
    }