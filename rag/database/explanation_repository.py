from typing import Any

from rag.database.supabase_client import get_supabase


def save_explanation(
    question: str,
    answer: str | None = None,
    simulation_context: dict | None = None,
    sources: list[dict[str, Any]] | None = None,
    confidence: float | None = None,
    session_id: str | None = None,
    experiment_id: str | None = None,
) -> dict[str, Any]:
    """
    Save a CARDIA explanation or grounded RAG context package
    to the Supabase explanations table.

    The answer is optional because the current RAG pipeline
    prepares grounded context before LLM answer generation.
    """

    row: dict[str, Any] = {
        "question": question,
        "answer": answer,
        "simulation_context": simulation_context,
        "sources": sources or [],
        "confidence": confidence,
    }

    if session_id:
        row["session_id"] = session_id

    if experiment_id:
        row["experiment_id"] = experiment_id

    response = (
        get_supabase()
        .table("explanations")
        .insert(row)
        .execute()
    )

    if not response.data:
        raise RuntimeError(
            "Supabase did not return the inserted explanation."
        )

    return response.data[0]