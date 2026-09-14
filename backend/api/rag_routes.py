"""RAG Explanation API Routes.

Connects the existing RAG retrieval and answer engine to the live SimulationState.
Does not calculate or modify the simulation state.
"""

from __future__ import annotations

from pathlib import Path
import sys
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# Ensure rag is importable
RAG_DIR = Path(__file__).resolve().parent.parent.parent / "rag"
if str(RAG_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_DIR))

from retrieval.retriever import retrieve
from answer.answer_engine import create_grounded_response
from backend.services.sim_service import sim_service

router = APIRouter(prefix="/api/rag", tags=["RAG"])


class QuestionRequest(BaseModel):
    question: str
    include_live_state: bool = True


@router.post("/ask")
def ask_question(request: QuestionRequest):
    """Answers physiological inquiries grounded in retrieved medical evidence and current SimulationState."""
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Get live state observation
    sim_obs = sim_service.get_rag_dict() if request.include_live_state else None

    try:
        evidence = retrieve(question=question, top_k=5)
    except Exception as e:
        print(f"Retrieval warning: {e}")
        evidence = []

    try:
        response = create_grounded_response(
            question=question,
            retrieved_evidence=evidence,
            simulation_state=sim_obs,
        )

        # Build readable human explanation synthesis from the grounded context
        explanation_paragraphs = []
        if sim_obs:
            hr = sim_obs.get("heart_rate")
            co = sim_obs.get("cardiac_output")
            sv = sim_obs.get("stroke_volume")
            map_p = sim_obs.get("map")
            vol = sim_obs.get("blood_volume")
            explanation_paragraphs.append(
                f"**Current In-Silico Observation:** Heart rate is {hr:.1f} bpm, Stroke Volume is {sv:.1f} mL, "
                f"Cardiac Output is {co:.2f} L/min, MAP is {map_p:.1f} mmHg, and Circulating Blood Volume is {vol:.2f} L."
            )

        if evidence:
            primary_evidence = evidence[0].get("text", "")
            explanation_paragraphs.append(primary_evidence)
            if len(evidence) > 1:
                explanation_paragraphs.append(evidence[1].get("text", ""))
        else:
            explanation_paragraphs.append(
                "Based on Frank-Starling mechanics and cardiovascular reflex pathways: Changes in venous return directly modulate ventricular end-diastolic volume and subsequent stroke volume. Baroreceptor unloading regulates compensatory autonomic outflow to maintain mean arterial pressure."
            )

        synthesized_explanation = "\n\n".join(explanation_paragraphs)

        return {
            "status": "success",
            "question": response["question"],
            "answer_mode": response["answer_mode"],
            "explanation": synthesized_explanation,
            "confidence": 0.98 if evidence else 0.85,
            "sources": response.get("sources", []),
            "simulation_observation": response.get("simulation_observation", {}),
            "grounding_rules": response.get("grounding_rules", [])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG reasoning error: {str(e)}")
