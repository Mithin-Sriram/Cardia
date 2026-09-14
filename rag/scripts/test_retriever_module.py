import sys
from pathlib import Path


# ============================================================
# PROJECT PATH
# ============================================================

RAG_DIR = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(RAG_DIR))


from retrieval.retriever import retrieve


# ============================================================
# TEST
# ============================================================

def main():

    question = "What happens if the AV node stops working?"

    print()
    print("=" * 60)
    print("CARDIA RETRIEVER FILTER TEST")
    print("=" * 60)

    print()
    print(f"Question: {question}")

    print()
    print("Filter:")
    print("Topic: cardiac_conduction")

    print()
    print("Retrieving evidence...")

    results = retrieve(
        question=question,
        top_k=5,
        topic="cardiac_conduction",
    )

    print()
    print("=" * 60)
    print("FILTERED RETRIEVED EVIDENCE")
    print("=" * 60)

    for index, result in enumerate(results, start=1):

        print()
        print(f"RESULT #{index}")
        print("-" * 60)

        print(f"Score: {result['score']:.4f}")
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"Title: {result['title']}")
        print(f"Source: {result['source']}")
        print(f"Topic: {result['topic']}")
        print(f"Organ: {result['organ']}")
        print(f"Mechanism: {result['mechanism']}")
        print(f"Equation: {result['equation']}")
        print(f"Page: {result['page']}")

        print()
        print("Text:")
        print(result["text"])

    print()
    print("=" * 60)
    print("FILTER TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()