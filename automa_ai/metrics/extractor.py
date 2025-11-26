from automa_ai.common.types import ModelMetrics


def extract_metrics_from_chunk(
    chunk,
    session_id=None,
    query_id=None
) -> ModelMetrics:

    rm = chunk.response_metadata or {}
    usage = chunk.usage_metadata or {}

    return ModelMetrics(
        session_id=session_id,
        query_id=query_id,

        model=rm.get("model") or rm.get("model_name") or "genai",
        model_provider=rm.get("model_provider"),
        created_at=rm.get("created_at"),

        total_duration=rm.get("total_duration"),
        load_duration=rm.get("load_duration"),
        prompt_eval_duration=rm.get("prompt_eval_duration"),
        eval_duration=rm.get("eval_duration"),

        prompt_eval_count=rm.get("prompt_eval_count"),
        eval_count=rm.get("eval_count"),

        input_tokens=usage.get("input_tokens"),
        output_tokens=usage.get("output_tokens"),
        total_tokens=usage.get("total_tokens"),

        raw_metadata={"response": rm, "usage": usage},
    )