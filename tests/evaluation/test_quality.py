import pytest
from src.engine.rag_pipeline import query_rag


def test_rag_quality_gate():
    golden_dataset = [
        {"query": "What is the refund policy?", "expected": "30 days"},
        # Add more samples
    ]

    pass_count = 0
    for item in golden_dataset:
        response = query_rag(item["query"])
        # Simplified string match or use an LLM-as-a-judge here
        if item["expected"].lower() in response.lower():
            pass_count += 1

    accuracy = pass_count / len(golden_dataset)

    # REGRESSION GATE: Fail the build if accuracy < 80%
    assert accuracy >= 0.8, f"Quality dropped to {accuracy}! Check the retriever."
