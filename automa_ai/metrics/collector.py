from collections import defaultdict
from typing import List

from automa_ai.common.types import ModelMetrics


class MetricsCollector:
    def __init__(self):
        self.records: List[ModelMetrics] = []
        self.current_query_id = None

    def add(self, metrics: ModelMetrics):
        self.records.append(metrics)

    def start_query(self, query_id: str):
        self.current_query_id = query_id

    def per_query(self):
        """Return metrics for the last completed query if agent resets between queries."""
        groups = defaultdict(list)
        for m in self.records:
            groups[m.query_id].append(m)
        return groups

    def per_session(self):
        """Return all metrics accumulated since the agent was created."""
        groups = defaultdict(list)
        for m in self.records:
            groups[m.session_id].append(m)
        return groups

    def summary_for_query(self, query_id: str):
        items = [m for m in self.records if m.query_id == query_id]
        return self._summarize(items)

    def summary_for_session(self, session_id: str):
        items = [m for m in self.records if m.session_id == session_id]
        return self._summarize(items)

    def _summarize(self, items):
        return {
            "num_calls": len(items),
            "models_used": list({m.model for m in items}),
            "tokens": {
                "input_tokens": sum(m.input_tokens or 0 for m in items),
                "output_tokens": sum(m.output_tokens or 0 for m in items),
                "total_tokens": sum((m.input_tokens or 0) + (m.output_tokens or 0) for m in items),
            },
            "durations": {
                "total_duration": sum(m.total_duration or 0 for m in items),
                "load_duration": sum(m.load_duration or 0 for m in items),
                "prompt_eval_duration": sum(m.prompt_eval_duration or 0 for m in items),
                "eval_duration": sum(m.eval_duration or 0 for m in items),
            }
        }