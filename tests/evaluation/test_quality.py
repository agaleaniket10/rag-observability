"""
RAG quality regression gate.

Tests the pipeline against a golden dataset and fails the build if
quality drops below the configured threshold.

NOTE: The pipeline currently uses a stub retriever and generator
(see src/engine/rag_pipeline.py). The golden dataset is calibrated
to the stub's output. When you wire in a real vector store and LLM,
update GOLDEN_DATASET to match your actual knowledge base content.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from src.engine.rag_pipeline import query_rag

# ── Thresholds (overridable via env vars) ─────────────────────────────────
ACCURACY_THRESHOLD = float(os.getenv("ACCURACY_THRESHOLD", "0.8"))

# ── Golden dataset ────────────────────────────────────────────────────────
# Keywords are checked case-insensitively with `in` — fast, free, and
# deterministic. No LLM judge needed for the CI gate.
#
# The stub pipeline always returns a fixed answer about RAG combining
# retrieval with a language model. All three queries below are answered
# by that response, so the gate passes at 100% with the stub and will
# catch regressions if the pipeline starts returning empty or broken output.
GOLDEN_DATASET = [
    {
        "query": "What is RAG?",
        "keywords": ["retrieval", "language model"],
        "description": "Basic RAG definition",
    },
    {
        "query": "How does RAG improve LLM accuracy?",
        "keywords": ["retriev", "accurate"],
        "description": "RAG accuracy improvement",
    },
    {
        "query": "What is retrieval-augmented generation?",
        "keywords": ["retrieval-augmented", "generation"],
        "description": "Full term match",
    },
]


# ── Helpers ───────────────────────────────────────────────────────────────


def _check_answer(answer: str, keywords: list[str]) -> bool:
    """Return True if ALL keywords appear in the answer (case-insensitive)."""
    answer_lower = answer.lower()
    return all(kw.lower() in answer_lower for kw in keywords)


def _save_report(results: list[dict], output_dir: str = "reports") -> None:
    """Write a JSON report for the CI artifact upload."""
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
                f"  Expected keywords: {item['keywords']}\n"
                f"  Got     : {answer[:300]}"
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
                    "answer_preview": answer[:200],
                }
            )

        _save_report(results)

        accuracy = pass_count / len(GOLDEN_DATASET)
        assert accuracy >= ACCURACY_THRESHOLD, (
            f"Quality gate FAILED: accuracy={accuracy:.0%} "
            f"(threshold={ACCURACY_THRESHOLD:.0%}). "
            f"See reports/quality_gate_results.json for details."
        )

    def test_answer_is_not_empty(self):
        """Smoke test — pipeline must return a non-empty string."""
        answer = query_rag("What is RAG?")
        assert isinstance(answer, str)
        assert len(answer.strip()) > 10, "Answer is suspiciously short"

    def test_pipeline_returns_string(self):
        """Type contract — query_rag must always return a str."""
        result = query_rag("test query")
        assert isinstance(result, str)
