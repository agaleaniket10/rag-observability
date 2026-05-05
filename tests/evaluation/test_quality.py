"""
RAG quality regression gate.

Runs the pipeline against a golden dataset and fails the build if
quality drops below configured thresholds.

The golden dataset is designed to match what the current pipeline
can answer — update it as you wire in a real vector store and LLM.
"""

from __future__ import annotations

import os
import json
from pathlib import Path

import pytest

from src.engine.rag_pipeline import query_rag

# ── Thresholds (overridable via env vars) ─────────────────────────────────
ACCURACY_THRESHOLD = float(os.getenv("ACCURACY_THRESHOLD", "0.8"))

# ── Golden dataset ────────────────────────────────────────────────────────
# Each entry has a query and one or more keywords that must appear in the
# answer. Keywords are lowercase and checked with `in` — no LLM judge needed
# for the CI gate (fast, free, deterministic).
GOLDEN_DATASET = [
    {
        "query": "What is RAG?",
        "keywords": ["retrieval", "language model"],
        "description": "Basic RAG definition",
    },
    {
        "query": "How does vector search work in RAG?",
        "keywords": ["embedding", "retriev"],
        "description": "Vector search explanation",
    },
    {
        "query": "What metrics are used to evaluate RAG?",
        "keywords": ["faithfulness", "relevancy", "recall"],
        "description": "RAG evaluation metrics",
    },
]


# ── Helpers ───────────────────────────────────────────────────────────────


def _check_answer(answer: str, keywords: list[str]) -> bool:
    """Return True if ALL keywords appear in the answer (case-insensitive)."""
    answer_lower = answer.lower()
    return all(kw.lower() in answer_lower for kw in keywords)


def _save_report(results: list[dict], output_dir: str = "reports") -> None:
    """Write a JSON report so the CI artifact upload has something to find."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    report_path = Path(output_dir) / "quality_gate_results.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)


# ── Tests ─────────────────────────────────────────────────────────────────


class TestRAGQualityGate:

    def test_individual_queries(self):
        """Each golden query must return an answer containing expected keywords."""
        for item in GOLDEN_DATASET:
            answer = query_rag(item["query"])
            assert isinstance(answer, str) and len(answer) > 0, f"Empty answer for: {item['query']}"
            assert _check_answer(answer, item["keywords"]), (
                f"FAIL [{item['description']}]\n"
                f"  Query   : {item['query']}\n"
                f"  Expected: {item['keywords']}\n"
                f"  Got     : {answer[:200]}"
            )

    def test_overall_accuracy_gate(self):
        """
        Regression gate — overall accuracy must stay above threshold.
        Fails the build if quality drops, catching regressions early.
        """
        results = []
        pass_count = 0

        for item in GOLDEN_DATASET:
            answer = query_rag(item["query"])
            passed = _check_answer(answer, item["keywords"])
            if passed:
                pass_count += 1
            results.append(
                {
                    "query": item["query"],
                    "description": item["description"],
                    "passed": passed,
                    "answer_preview": answer[:150],
                }
            )

        _save_report(results)

        accuracy = pass_count / len(GOLDEN_DATASET)
        assert accuracy >= ACCURACY_THRESHOLD, (
            f"Quality gate FAILED: accuracy={accuracy:.0%} "
            f"(threshold={ACCURACY_THRESHOLD:.0%}). "
            f"Check reports/quality_gate_results.json for details."
        )

    def test_answer_is_not_empty(self):
        """Smoke test — pipeline must return non-empty strings."""
        answer = query_rag("What is RAG?")
        assert isinstance(answer, str)
        assert len(answer.strip()) > 10, "Answer is suspiciously short"

    def test_pipeline_returns_string(self):
        """Type contract — query_rag must always return a str."""
        result = query_rag("test query")
        assert isinstance(result, str)
