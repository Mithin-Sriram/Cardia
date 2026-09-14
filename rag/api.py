from pathlib import Path
import sys

from fastapi import FastAPI
from pydantic import BaseModel


# --------------------------------------------------
# RAG DIRECTORY
# --------------------------------------------------

RAG_DIR = Path(__file__).resolve().parent

if str(RAG_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_DIR))


# --------------------------------------------------
# RAG COMPONENTS
# --------------------------------------------------

from retrieval.retriever import retrieve
from answer.answer_engine import create_grounded_response

from rag.database.explanation_repository import save_explanation


# --------------------------------------------------
# FASTAPI APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="CARDIA RAG API",
    description=(
        "Cardiology-focused retrieval and grounded "
        "answer engine for CARDIA."
    ),
    version="1.0.0",
)


# --------------------------------------------------
# REQUEST MODEL
# --------------------------------------------------

class QuestionRequest(BaseModel):
    question: str
    simulation_state: dict | None = None


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "cardia-rag",
    }


# --------------------------------------------------
# ASK ENDPOINT
# --------------------------------------------------

@app.post("/ask")
def ask_question(request: QuestionRequest):
    question = request.question.strip()

    if not question:
        return {
            "status": "error",
            "message": "Question cannot be empty.",
        }


    # --------------------------------------------------
    # RETRIEVE PHYSIOLOGY EVIDENCE
    # --------------------------------------------------

    evidence = retrieve(
        question=question,
        top_k=5,
    )


    # --------------------------------------------------
    # BUILD GROUNDED RESPONSE
    # --------------------------------------------------

    response = create_grounded_response(
        question=question,
        retrieved_evidence=evidence,
        simulation_state=request.simulation_state,
    )


    # --------------------------------------------------
    # PERSIST GROUNDED CONTEXT
    # --------------------------------------------------

    saved_explanation = save_explanation(
        question=response["question"],
        answer=None,
        simulation_context=response[
            "simulation_observation"
        ],
        sources=response["sources"],
        confidence=None,
    )


    # --------------------------------------------------
    # RETURN RESPONSE
    # --------------------------------------------------

    response["database"] = {
        "saved": True,
        "explanation_id": saved_explanation["id"],
    }

    return response