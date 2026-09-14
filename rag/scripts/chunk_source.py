from pathlib import Path
import json
import re


INPUT_FILE = Path("data/clean/cardia_physiology.txt")
OUTPUT_FILE = Path("data/chunks/cardia_chunks.json")


def normalize_title(line: str):
    """
    Convert a numbered section heading into its clean title.

    Example:
        1. Heart Rate
    becomes:
        Heart Rate
    """

    line = line.strip()

    line = re.sub(
        r"^\d+\.\s*",
        "",
        line,
    )

    return line.strip()


def is_main_section_title(line: str):
    """
    Detect actual CARDIA knowledge-base section headings.

    The source contains numbered physiological sections,
    but some sections also contain numbered reasoning rules.
    Only the known main headings are treated as boundaries.
    """

    main_titles = {
        "CARDIA Simulation Variables",
        "Heart Rate",
        "Stroke Volume",
        "Cardiac Output",
        "EDV",
        "ESV",
        "Cardiac Conduction System",
        "SA Node",
        "AV Node",
        "Bundle of His",
        "Purkinje Fibers",
        "Cardiac Valves",
        "Mitral Valve",
        "Aortic Valve",
        "Tricuspid Valve",
        "Pulmonary Valve",
        "Blood Pressure",
        "Mean Arterial Pressure",
        "Systemic Vascular Resistance",
        "Left Ventricular Pressure",
        "Aortic Pressure",
        "Contractility",
        "Blood Volume",
        "Cardiac Cycle",
        "Cause-and-Effect Reasoning Rules",
        "Current CARDIA Baseline State",
        "Evidence Sources",
        "RAG Safety Rules",
        "Source Provenance",
    }

    clean_line = normalize_title(line)

    return clean_line in main_titles


def split_sections(text: str):
    """
    Split the cleaned CARDIA physiology document into
    meaningful physiological sections.

    Numbered sub-rules inside a section remain part of
    that section instead of becoming separate chunks.
    """

    lines = text.splitlines()

    sections = []

    current_title = None
    current_content = []

    for line in lines:

        line = line.strip()

        if not line:
            continue

        if is_main_section_title(line):

            if current_title is not None:

                sections.append(
                    {
                        "title": current_title,
                        "text": "\n".join(
                            current_content
                        ).strip(),
                    }
                )

            current_title = normalize_title(line)

            current_content = []

        else:

            current_content.append(line)

    if current_title is not None:

        sections.append(
            {
                "title": current_title,
                "text": "\n".join(
                    current_content
                ).strip(),
            }
        )

    return sections


def determine_topic(title: str, text: str):
    """
    Assign a single retrieval topic based primarily
    on the section title.
    """

    title_lower = title.lower()

    title_topics = {

        "heart rate":
            "heart_rate",

        "stroke volume":
            "cardiac_output",

        "cardiac output":
            "cardiac_output",

        "end-diastolic volume":
            "cardiac_output",

        "end-systolic volume":
            "cardiac_output",

        "cardiac conduction system":
            "cardiac_conduction",

        "sa node":
            "cardiac_conduction",

        "av node":
            "cardiac_conduction",

        "bundle of his":
            "cardiac_conduction",

        "purkinje fibers":
            "cardiac_conduction",

        "cardiac valves":
            "cardiac_valves",

        "mitral valve":
            "cardiac_valves",

        "aortic valve":
            "cardiac_valves",

        "tricuspid valve":
            "cardiac_valves",

        "pulmonary valve":
            "cardiac_valves",

        "blood pressure":
            "hemodynamics",

        "mean arterial pressure":
            "hemodynamics",

        "systemic vascular resistance":
            "hemodynamics",

        "left ventricular pressure":
            "hemodynamics",

        "aortic pressure":
            "hemodynamics",

        "contractility":
            "contractility",

        "blood volume":
            "blood_volume",

        "cardiac cycle":
            "cardiac_cycle",

        "cause-and-effect reasoning rules":
            "reasoning",

        "current cardia baseline state":
            "simulation_state",

        "evidence sources":
            "provenance",

        "rag safety rules":
            "safety",

        "source provenance":
            "provenance",
    }

    if title_lower in title_topics:
        return title_topics[title_lower]

    if title_lower == "cardia simulation variables":
        return "simulation_state"

    if title_lower == "edv":
        return "cardiac_output"

    if title_lower == "esv":
        return "cardiac_output"

    return "general_physiology"


def determine_mechanisms(
    title: str,
    text: str,
):
    """
    Detect physiological mechanisms relevant to
    retrieval and future metadata filtering.
    """

    combined = f"{title} {text}".lower()

    mechanism_keywords = {

        "electrical_conduction": [
            "electrical",
            "conduction",
            "impulse",
            "pacemaker",
            "depolarization",
        ],

        "preload": [
            "preload",
            "edv",
            "venous return",
            "filling",
        ],

        "afterload": [
            "afterload",
            "svr",
            "aortic pressure",
        ],

        "contractility": [
            "contractility",
            "inotropy",
        ],

        "pressure": [
            "pressure",
            "blood pressure",
            "map",
        ],

        "valve_flow": [
            "valve",
            "flow",
            "regurgitation",
            "opening",
            "closing",
        ],

        "ventricular_ejection": [
            "ejection",
            "ventricular ejection",
        ],

        "ventricular_filling": [
            "filling",
            "ventricular filling",
        ],
    }

    mechanisms = []

    for mechanism_name, keywords in mechanism_keywords.items():

        if any(
            keyword in combined
            for keyword in keywords
        ):

            mechanisms.append(
                mechanism_name
            )

    return mechanisms


def extract_equations(text: str):
    """
    Detect known CARDIA physiology equations
    present inside each chunk.
    """

    equations = []

    known_equations = [

        "CO = HR × SV",

        "SV = EDV - ESV",

        "MAP ≈ DBP + 1/3(SBP - DBP)",

        "MAP = (SBP + 2 × DBP) / 3",
    ]

    for equation in known_equations:

        if equation in text:

            equations.append(
                equation
            )

    return equations


def infer_metadata(
    title: str,
    text: str,
):
    """
    Build the metadata schema used by the RAG system.
    """

    return {

        "source":
            "cardia_physiology.md",

        "topic":
            determine_topic(
                title,
                text,
            ),

        "organ":
            "heart",

        "mechanism":
            determine_mechanisms(
                title,
                text,
            ),

        "equation":
            extract_equations(
                text,
            ),

        "page":
            None,
    }


def main():

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    text = INPUT_FILE.read_text(
        encoding="utf-8"
    )

    sections = split_sections(text)

    if len(sections) <= 1:

        raise ValueError(
            "The physiology source was not split into "
            "multiple meaningful sections."
        )

    chunks = []

    for index, section in enumerate(
        sections
    ):

        metadata = infer_metadata(
            section["title"],
            section["text"],
        )

        chunk = {

            "chunk_id":
                f"cardia_{index + 1:03d}",

            "title":
                section["title"],

            "text":
                section["text"],

            "metadata":
                metadata,
        }

        chunks.append(chunk)

    OUTPUT_FILE.write_text(

        json.dumps(
            chunks,
            indent=2,
            ensure_ascii=False,
        ),

        encoding="utf-8",
    )

    print(
        "Chunking complete."
    )

    print(
        f"Input: {INPUT_FILE}"
    )

    print(
        f"Output: {OUTPUT_FILE}"
    )

    print(
        f"Chunks created: {len(chunks)}"
    )

    print(
        "\nAll chunks:"
    )

    for chunk in chunks:

        print(
            f"[{chunk['chunk_id']}] "
            f"{chunk['title']} "
            f"→ "
            f"{chunk['metadata']['topic']}"
        )


if __name__ == "__main__":

    main()