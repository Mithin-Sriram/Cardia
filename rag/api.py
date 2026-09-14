from pathlib import Path
import sys

from fastapi import FastAPI
from pydantic import BaseModel

RAG_DIR = Path(__file__).resolve().parent

if str(RAG_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_DIR))

from retrieval.retriever import retrieve
from answer.answer_engine import create_grounded_response
from rag.database.explanation_repository import save_explanation
from rag.database.session_repository import get_session


app = FastAPI(
    title="CARDIA RAG API",
    description=(
        "Cardiology-focused retrieval and grounded "
        "answer engine for CARDIA."
    ),
    version="1.0.0",
)


class QuestionRequest(BaseModel):
    question: str
    simulation_state: dict | None = None
    session_id: str | None = None


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "cardia-rag",
    }


@app.post("/ask")
def ask_question(request: QuestionRequest):
    question = request.question.strip()

    if not question:
        return {
            "status": "error",
            "message": "Question cannot be empty.",
        }

    session = None

    if request.session_id:
        session = get_session(request.session_id)

    evidence = retrieve(
        question=question,
        top_k=5,
    )

    response = create_grounded_response(
        question=question,
        retrieved_evidence=evidence,
        simulation_state=request.simulation_state,
    )

    saved_explanation = save_explanation(
        question=response["question"],
        answer=None,
        simulation_context=response[
            "simulation_observation"
        ],
        sources=response["sources"],
        confidence=None,
        session_id=request.session_id,
    )

    response["database"] = {
        "saved": True,
        "explanation_id": saved_explanation["id"],
        "session_id": request.session_id,
    }

    if session:
        response["database"]["session"] = {
            "id": session["id"],
            "status": session["status"],
        }

    return response