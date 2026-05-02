import time
from dataclasses import dataclass


@dataclass
class RequestMetrics:
    latency: float
    input_tokens: int
    output_tokens: int
    cost: float


class ObservabilityManager:
    # Approximate pricing for GPT-4o-mini
    COST_PER_1K_INPUT = 0.00015
    COST_PER_1K_OUTPUT = 0.0006

    def calculate_cost(self, input_tokens, output_tokens):
        return (input_tokens / 1000 * self.COST_PER_1K_INPUT) + (
            output_tokens / 1000 * self.COST_PER_1K_OUTPUT
        )

    def log_request(self, start_time, usage):
        latency = time.time() - start_time
        cost = self.calculate_cost(usage.prompt_tokens, usage.completion_tokens)

        # In a real app, push these to Prometheus or a DB
        print(f"📊 Latency: {latency:.2f}s | Cost: ${cost:.6f}")
        return RequestMetrics(
            latency, usage.prompt_tokens, usage.completion_tokens, cost
        )
