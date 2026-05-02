"""
Token-to-USD cost tracking for LLM calls in the RAG pipeline.
"""

from dataclasses import dataclass, field
from typing import Dict, Union

# Pricing per 1,000 tokens (USD) — update as provider pricing changes
MODEL_PRICING: Dict[str, Dict[str, float]] = {
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    "claude-3-5-sonnet": {"input": 0.003, "output": 0.015},
    "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
}


@dataclass
class UsageRecord:
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float = field(init=False)

    def __post_init__(self) -> None:
        self.cost_usd = calculate_cost(
            self.model, self.input_tokens, self.output_tokens
        )


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """
    Convert token counts to USD based on known model pricing.
    Returns 0.0 if the model is not in the pricing table.
    """
    pricing = MODEL_PRICING.get(model)
    if not pricing:
        return 0.0

    input_cost = (input_tokens / 1_000) * pricing["input"]
    output_cost = (output_tokens / 1_000) * pricing["output"]
    return round(input_cost + output_cost, 6)


class CostTracker:
    """Accumulates cost across multiple LLM calls within a session."""

    def __init__(self) -> None:
        self._records: list[UsageRecord] = []

    def record(self, model: str, input_tokens: int, output_tokens: int) -> UsageRecord:
        record = UsageRecord(
            model=model, input_tokens=input_tokens, output_tokens=output_tokens
        )
        self._records.append(record)
        return record

    @property
    def total_cost_usd(self) -> float:
        return round(sum(r.cost_usd for r in self._records), 6)

    @property
    def total_input_tokens(self) -> int:
        return sum(r.input_tokens for r in self._records)

    @property
    def total_output_tokens(self) -> int:
        return sum(r.output_tokens for r in self._records)

    def summary(self) -> Dict[str, Union[float, int]]:
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": self.total_cost_usd,
            "num_calls": len(self._records),
        }

    def reset(self) -> None:
        self._records.clear()
