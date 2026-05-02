"""
Entry point for the RAG observability service.
"""

import os
from dotenv import load_dotenv

from src.monitor.tracing import setup_tracing
from src.engine.rag_pipeline import RAGPipeline

load_dotenv()


def main() -> None:
    # Enable LangSmith tracing if configured
    if os.getenv("LANGCHAIN_API_KEY"):
        setup_tracing()

    pipeline = RAGPipeline(model=os.getenv("LLM_MODEL", "gpt-4o-mini"))

    # Example query — replace with your application logic or API server
    question = "What is retrieval-augmented generation?"
    result = pipeline.query(question)

    print(f"Answer : {result['answer']}")
    print(f"Sources: {result['sources']}")
    print(f"Cost   : ${result['cost_usd']:.6f}")
    print(f"Session cost summary: {pipeline.cost_tracker.summary()}")


if __name__ == "__main__":
    main()
