"""
Core RAG pipeline — retrieval + generation with integrated observability hooks.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

from src.monitor.metrics import ObservabilityManager
from src.monitor.cost_tracker import CostTracker


class RAGPipeline:
    """
    Minimal RAG pipeline skeleton.

    Replace the stub methods with your actual vector store and LLM client.
    """

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self.model = model
        self.cost_tracker = CostTracker()
        self.observability = ObservabilityManager()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def query(self, question: str) -> Dict[str, Any]:
        """
        Run a full RAG cycle: retrieve relevant docs, then generate an answer.

        Returns a dict with keys: answer, sources, cost_usd.
        """
        start_time = time.time()

        docs = self._retrieve(question)
        answer, usage = self._generate(question, docs)

        # Log latency and cost via ObservabilityManager
        metrics = self.observability.log_request(start_time, usage)

        # Also accumulate cost in the session-level CostTracker
        self.cost_tracker.record(
            model=self.model,
            input_tokens=metrics.input_tokens,
            output_tokens=metrics.output_tokens,
        )

        return {
            "answer": answer,
            "sources": [d["id"] for d in docs],
            "cost_usd": metrics.cost,
        }

    # ------------------------------------------------------------------
    # Private stubs — replace with real implementations
    # ------------------------------------------------------------------

    def _retrieve(self, question: str) -> List[Dict[str, Any]]:
        """Retrieve relevant documents from the vector store."""
        # TODO: replace with actual vector store call
        # e.g. self.vector_store.similarity_search(question, k=5)
        return [
            {
                "id": "doc-1",
                "content": "Retrieval-augmented generation (RAG) combines a retrieval system with a generative language model to produce grounded, accurate answers.",
            },
            {
                "id": "doc-2",
                "content": "Vector stores enable semantic search by encoding documents as dense embeddings and retrieving the top-k most similar results at query time.",
            },
            {
                "id": "doc-3",
                "content": "Common RAG evaluation metrics include faithfulness, answer relevancy, and context recall.",
            },
        ]

    def _generate(
        self, question: str, context_docs: List[Dict[str, Any]]
    ) -> tuple[str, Any]:
        """
        Generate an answer from the LLM given retrieved context.

        Returns (answer, usage) where usage exposes .prompt_tokens and
        .completion_tokens to match the ObservabilityManager.log_request
        interface.
        """
        context = "\n\n".join(d.get("content", "") for d in context_docs)
        prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"

        # Simulate realistic LLM response latency
        time.sleep(1.2)

        # TODO: replace with actual LLM call, e.g.:
        # response = openai_client.chat.completions.create(...)
        # return response.choices[0].message.content, response.usage

        answer = (
            "Retrieval-augmented generation (RAG) is a technique that enhances "
            "a language model by retrieving relevant documents from a knowledge base "
            "and using them as context during generation, resulting in more accurate "
            "and grounded responses."
        )
        usage = _StubUsage(prompt_tokens=512, completion_tokens=128)
        return answer, usage


class _StubUsage:
    """Mimics the shape of openai.types.CompletionUsage for the stub path."""

    def __init__(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


def query_rag(question: str) -> str:
    """
    Module-level convenience wrapper used by the evaluation test suite.
    """
    pipeline = RAGPipeline()
    result = pipeline.query(question)
    return result["answer"]
