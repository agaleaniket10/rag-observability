import os
from langsmith import Client


def setup_tracing():
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "RAG-Obs-Project"
    # Ensure LANGCHAIN_API_KEY is in your .env
    return Client()
