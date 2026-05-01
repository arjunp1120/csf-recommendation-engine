from csf_recommendation_engine.jobs.weekend_validation import build_weekend_validation_summary


def test_weekend_validation_summary_is_blocked_until_decisions_are_made() -> None:
    summary = build_weekend_validation_summary()

    assert summary["status"] == "blocked"
    assert "holdout dataset" in summary["detail"]
    assert "serialization trigger path" in summary["detail"]
