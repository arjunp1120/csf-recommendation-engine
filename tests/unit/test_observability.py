import logging

from csf_recommendation_engine.core.logging import JsonLogFormatter
from csf_recommendation_engine.core.observability import MetricsRecorder


def test_metrics_recorder_tracks_counters_and_timings() -> None:
    recorder = MetricsRecorder()
    recorder.increment("recommend_requests")
    recorder.record_timing("recommend_latency", 0.25)

    snapshot = recorder.snapshot()

    assert snapshot["counters"]["recommend_requests"] == 1
    assert snapshot["timings"]["recommend_latency"] == [0.25]


def test_json_log_formatter_includes_request_context_fields() -> None:
    formatter = JsonLogFormatter()
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello",
        args=(),
        exc_info=None,
    )
    record.request_id = "req-1"
    record.model_id = "model-1"

    payload = formatter.format(record)

    assert '"request_id": "req-1"' in payload
    assert '"model_id": "model-1"' in payload
