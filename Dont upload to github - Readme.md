Think of building a standard RAG system like opening a **new restaurant**. Most people focus only on the recipe (the AI's answer). This project is about building the **kitchen management system** that ensures the restaurant doesn't go bankrupt or accidentally poison the customers.

Here is how the four technical pillars translate to running that restaurant:

---

### 1. Tracing: The "Security Camera"
Imagine a customer complains their soup is cold. Without tracing, you just know the customer is unhappy. With **Tracing**, you have a camera in the kitchen. You can see exactly where the delay happened:
*   Did the waiter take too long to deliver it? (The Retrieval)
*   Or did the chef take too long to cook it? (The LLM Generation)
*   **The Benefit:** You don't guess; you know exactly which part of the process is "broken."

### 2. Latency (p50/p95): The "Service Guarantee"
If you tell a customer their food takes "10 minutes on average," that’s misleading. If nine people get their food in 2 minutes, but the tenth person waits 2 hours, the average looks okay, but that tenth person is never coming back.
*   **p50:** The time a "typical" customer waits.
*   **p95:** The "worst-case" scenario (the slowest 5% of orders).
*   **The Benefit:** You optimize for the slow experiences so *nobody* feels forgotten.

### 3. Cost-per-Request: The "Ingredient Tracker"
In an AI system, every word the AI reads or writes costs a fraction of a penny. If you aren't careful, a single complex question could cost you $2.00 to answer, while you only charged the customer $0.50.
*   **The Benefit:** It’s like a digital scale on the chef's counter. It tells you exactly how much money in "ingredients" (tokens) just walked out the door for every single plate served.

### 4. Regression Gating: The "Head Chef’s Taste Test"
Imagine you decide to change the salt in your recipe. You might think it tastes better, but does it ruin the dish for everyone else? 
*   **The Gate:** Before any new recipe (code) is allowed on the menu, it has to pass a "Taste Test" against 100 classic dishes you know must be perfect. If the new salt makes even 20 of those dishes taste bad, the kitchen "gates" the change and refuses to serve it.
*   **The Benefit:** You can upgrade your system without worrying that you’ve accidentally made it "stupid" in ways you didn't notice.

---

### Why this matters:
Most people just build a "kitchen" and hope for the best. By adding these four things, you’ve built a **business**. You’re showing that you care about speed, money, and quality control—which is exactly what happens in the real world.



Technical explanation:


Moving back into the engineering specifics, these four pillars are implemented by hooking into the lifecycle of an LLM call. In a production environment, you don't just "run" a prompt; you wrap it in **middleware** and **interceptors**.

---

## 1. Distributed Tracing (The Execution Path)
In a RAG system, "Tracing" means following a unique `trace_id` as it moves from the user's query, through the embedding model, into the vector database, and finally through the LLM. 

We use **OpenTelemetry** standards so that if the system lags, we can see if the bottleneck was the database lookup or the model generation.

```python
from langsmith import traceable

# This decorator automatically logs inputs, outputs, and internal timing
@traceable(run_type="retriever")
def retrieve_docs(query):
    # Logic to fetch from Vector DB (Chroma/Pinecone)
    return docs

@traceable(run_type="llm")
def generate_answer(query, context):
    # Logic for LLM call
    return response
```

---

## 2. Latency Metrics (p50/p95 Distribution)
We don't care about the arithmetic mean (average) because LLM latency is "jittery." We track **percentiles**. 
*   **p50 (Median):** Half of your users experience this speed.
*   **p95:** The threshold that 95% of requests stay under. If p95 spikes, your "tail latency" is hurting your power users.



```python
import time
import numpy as np

latencies = []

def track_latency(start_time):
    duration = time.time() - start_time
    latencies.append(duration)
    
    # Calculate percentiles using numpy
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    return p50, p95
```

---

## 3. Cost-per-Request (Unit Economics)
To track costs, we must intercept the `usage` metadata returned by providers (like OpenAI or Anthropic). We map these token counts against a price table.

```python
# Standardized Pricing Map
PRICES = {
    "gpt-4o": {"input": 0.005 / 1000, "output": 0.015 / 1000},
    "gpt-4o-mini": {"input": 0.00015 / 1000, "output": 0.0006 / 1000}
}

def calculate_request_cost(model, usage):
    input_cost = usage.prompt_tokens * PRICES[model]["input"]
    output_cost = usage.completion_tokens * PRICES[model]["output"]
    total_cost = input_cost + output_cost
    return total_cost
```

---

## 4. Regression Gating (CI/CD Quality Control)
This is the most critical part of AI-Ops. A **Regression Gate** is a test suite that runs in your GitHub Actions. It uses **RAGAS** (Retrieval-Augmented Generation Assessment) to score the system on:
1.  **Faithfulness:** Did the LLM hallucinate or stick to the docs?
2.  **Relevancy:** Did the retriever find the *right* docs?



```python
from ragas import evaluate
from datasets import Dataset

def test_production_quality():
    # 1. Create a "Golden Dataset" of questions and ground truths
    data_sample = {
        "question": ["How do I reset my password?"],
        "contexts": [["Users can reset passwords via the settings tab..."]],
        "answer": ["You can reset your password in the settings tab."],
        "ground_truth": ["Go to settings to reset your password."]
    }
    dataset = Dataset.from_dict(data_sample)
    
    # 2. Run the evaluation
    score = evaluate(dataset)
    
    # 3. The Gate: Fail the build if the score is too low
    assert score['faithfulness'] > 0.85 
```

