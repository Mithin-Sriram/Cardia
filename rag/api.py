from pathlib import Path
import sys

from fastapi import FastAPI
from pydantic import BaseModel

RAG_DIR = Path(__file__).resolve().parent

if str(RAG_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_DIR))

from retrieval.retriever import retrieve
from answer.answer_engine import create_grounded_response
from rag.llm.gemini_generator import generate_grounded_answer
from rag.answer.confidence import (
    calculate_confidence,
    confidence_level,
)
from rag.database.explanation_repository import (
    save_explanation,
    get_session_explanations,
)
from rag.database.session_repository import get_session
from rag.database.knowledge_source_repository import (
    find_knowledge_source,
)


app = FastAPI(
    title="CARDIA RAG API",
    description=(
        "Cardiology-focused retrieval, grounded reasoning, "
        "LLM generation, confidence scoring, session-aware "
        "conversation context, and persistence API for CARDIA."
    ),
    version="1.3.0",
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
    session_history = []

    if request.session_id:
        session = get_session(request.session_id)

        session_history = get_session_explanations(
            session_id=request.session_id,
            limit=10,
        )

    evidence = retrieve(
        question=question,
        top_k=5,
    )

    response = create_grounded_response(
        question=question,
        retrieved_evidence=evidence,
        simulation_state=request.simulation_state,
    )

    confidence = calculate_confidence(
        response["evidence"]
    )

    confidence_label = confidence_level(
        confidence
    )

    response["confidence"] = confidence
    response["confidence_level"] = confidence_label

    knowledge_sources = []

    for source in response["sources"]:
        knowledge_source = find_knowledge_source(
            title=source.get("source_title", ""),
            author=source.get("author"),
            source_type=source.get("source_type"),
        )

        if knowledge_source:
            source["knowledge_source_id"] = (
                knowledge_source["id"]
            )

            knowledge_source_summary = {
                "id": knowledge_source["id"],
                "title": knowledge_source["title"],
                "author": knowledge_source["author"],
                "source_type": knowledge_source["source_type"],
            }

            if (
                knowledge_source_summary
                not in knowledge_sources
            ):
                knowledge_sources.append(
                    knowledge_source_summary
                )

    answer = generate_grounded_answer(
        question=question,
        grounded_response=response,
        session_history=session_history,
    )

    saved_explanation = save_explanation(
        question=response["question"],
        answer=answer,
        simulation_context=response[
            "simulation_observation"
        ],
        sources=response["sources"],
        confidence=confidence,
        session_id=request.session_id,
    )

    response["answer"] = answer

    response["knowledge_sources"] = knowledge_sources

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

    response["session_context"] = {
        "used": bool(session_history),
        "exchange_count": len(session_history),
    }

    return response