import os
from typing import Any

from dotenv import load_dotenv
from google import genai


RAG_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

load_dotenv(os.path.join(RAG_DIR, ".env"))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is missing from rag/.env"
    )


client = genai.Client(
    api_key=GEMINI_API_KEY
)

MODEL_NAME = "gemini-3.6-flash"


SYSTEM_INSTRUCTIONS = """
You are CARDIA, a cardiology education assistant
for medical students.

Your job is to explain cardiovascular physiology
using ONLY the grounded evidence supplied to you.

IMPORTANT RULES:

1. Treat the retrieved evidence as the primary
   knowledge source.

2. Treat SimulationState values as observations
   from the CARDIA simulation.

3. Never invent simulation variables, measurements,
   events, trends, or simulation results.

4. Never claim that CARDIA simulated a hypothetical
   situation unless an actual simulation result is
   explicitly provided.

5. Clearly distinguish:
   - established physiological knowledge
   - observations from the simulation
   - hypothetical physiological reasoning

6. If the supplied evidence is insufficient to answer
   something confidently, say so.

7. Do not fabricate citations, references, sources,
   measurements, or physiological findings.

8. Explain concepts at a medical-student level:
   accurate, clear, structured, and concise.

9. When discussing a physiological mechanism,
   explain the cause-and-effect relationship.

10. The answer must remain grounded in the supplied
    context rather than relying on unsupported memory.

11. Do not describe a simulation state as "stable",
    "normal", "abnormal", "healthy", "pathological",
    "baseline", or any similar clinical classification
    unless that conclusion is explicitly supported by
    the supplied evidence or an explicit simulation
    result.

12. Do not infer clinical diagnoses or patient conditions
    from isolated simulation values.

13. Do not interpret a numerical value as clinically
    normal or abnormal unless an appropriate reference
    range or explicit evidence is supplied.

14. When numerical values are provided, report them as
    observations first. Then explain their physiological
    meaning only when supported by the retrieved evidence.

15. Do not imply causation merely because two simulation
    variables appear together. Distinguish observed
    relationships from established physiological
    mechanisms.

16. For hypothetical questions, use language such as
    "would tend to", "would be expected to", or
    "physiologically" when appropriate.

17. If a question asks what CARDIA currently shows,
    use only the supplied SimulationState values for
    current-state claims.

18. If the evidence does not support a conclusion,
    explicitly state that the available information
    is insufficient.
"""


def _format_evidence(
    evidence: list[dict[str, Any]],
) -> str:
    """
    Format the actual retrieved evidence chunks
    for Gemini.

    This uses the 'evidence' list from the Answer Engine,
    which contains the physiological text.
    """

    if not evidence:
        return "No retrieved evidence."

    formatted = []

    for index, item in enumerate(
        evidence,
        start=1,
    ):
        formatted.append(
            f"""
EVIDENCE {index}
Chunk ID: {item.get("chunk_id")}
Title: {item.get("title")}
Source: {item.get("source_title")}
Author: {item.get("author")}
Chapter: {item.get("chapter")}
Topic: {item.get("topic")}
Organ: {item.get("organ")}
Source type: {item.get("source_type")}
Authority: {item.get("authority_level")}
Page: {item.get("page")}
Similarity score: {item.get("score")}

Physiological evidence:
{item.get("text") or "No text available."}

Mechanisms:
{item.get("mechanism", [])}

Equations:
{item.get("equation", [])}
""".strip()
        )

    return "\n\n".join(formatted)


def _format_simulation(
    simulation_observation: dict[str, Any],
) -> str:
    if not simulation_observation.get("available"):
        return (
            "No SimulationState was supplied. "
            "Do not describe current simulation values."
        )

    values = simulation_observation.get(
        "values",
        {},
    )

    if not values:
        return (
            "SimulationState was supplied but contains "
            "no usable values."
        )

    lines = [
        "Current CARDIA SimulationState observations:"
    ]

    for key, value in values.items():
        lines.append(
            f"- {key}: {value}"
        )

    return "\n".join(lines)


def generate_grounded_answer(
    question: str,
    grounded_response: dict[str, Any],
) -> str:
    """
    Generate a student-facing answer from the already
    grounded CARDIA response package.

    Gemini receives the actual retrieved evidence and
    validated simulation observations.

    Gemini does not control or modify the simulation.
    """

    evidence = grounded_response.get(
        "evidence",
        [],
    )

    simulation_observation = grounded_response.get(
        "simulation_observation",
        {},
    )

    answer_mode = grounded_response.get(
        "answer_mode",
        "physiology",
    )

    prompt = f"""
{SYSTEM_INSTRUCTIONS}

ANSWER MODE:
{answer_mode}

STUDENT QUESTION:
{question}

RETRIEVED CARDIOLOGY EVIDENCE:
{_format_evidence(evidence)}

SIMULATION OBSERVATION:
{_format_simulation(simulation_observation)}

GROUNDING REQUIREMENTS:

- Answer the student's question directly.
- Use the retrieved evidence as the factual basis.
- If SimulationState is unavailable, do not invent
  current CARDIA values.
- If this is a hypothetical question, explicitly
  identify the explanation as physiological reasoning
  rather than an actual simulation result.
- If this is a simulation question, clearly separate
  observed values from physiological interpretation.
- Report simulation values as observations unless
  supporting evidence justifies further interpretation.
- Do not label a simulation state as stable, normal,
  abnormal, healthy, pathological, or baseline unless
  the supplied context explicitly supports that label.
- Do not infer a diagnosis from the simulation values.
- Do not mention internal software implementation
  unless it is relevant to the question.
- Do not create fake references.
- Keep the answer focused and medically accurate.
- Do not introduce physiological facts that are absent
  from the retrieved evidence unless they are directly
  necessary to interpret the supplied evidence.
- If the retrieved evidence is insufficient, say so.
- For hypothetical reasoning, distinguish expected
  physiological effects from actual CARDIA results.

Now provide the final student-facing answer.
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )

    if not response.text:
        raise RuntimeError(
            "Gemini returned an empty response."
        )

    return response.text.strip()