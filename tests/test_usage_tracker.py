"""Tests for LLM usage tracking."""


def test_log_call_and_query(tmp_path):
    from cairn.usage_tracker import UsageTracker

    tracker = UsageTracker(db_path=str(tmp_path / "usage.db"))
    tracker.log_call(
        session_id="sess-123",
        tool_name="cairn_store",
        model="claude-opus-4-6",
        input_tokens=1000,
        output_tokens=500,
        project="test",
    )
    usage = tracker.get_usage(days=1, group_by="model")
    assert len(usage) == 1
    assert usage[0]["model"] == "claude-opus-4-6"
    assert usage[0]["total_input_tokens"] == 1000
    assert usage[0]["total_output_tokens"] == 500
    tracker.close()


def test_cost_estimation(tmp_path):
    from cairn.usage_tracker import UsageTracker

    tracker = UsageTracker(db_path=str(tmp_path / "usage.db"))
    tracker.log_call(
        session_id="sess-123",
        tool_name="cairn_query",
        model="claude-opus-4-6",
        input_tokens=1_000_000,
        output_tokens=100_000,
    )
    cost = tracker.get_cost_estimate(days=30)
    # Opus: 15/M input + 75/M output = $15 + $7.50 = $22.50
    assert cost["total_usd"] > 20
    assert cost["total_usd"] < 25
    tracker.close()


def test_top_tools(tmp_path):
    from cairn.usage_tracker import UsageTracker

    tracker = UsageTracker(db_path=str(tmp_path / "usage.db"))
    for i in range(5):
        tracker.log_call("s1", "cairn_store", "claude-sonnet-4-6", 100, 50)
    for i in range(2):
        tracker.log_call("s1", "cairn_query", "claude-sonnet-4-6", 200, 100)

    top = tracker.get_top_tools(days=1, limit=5)
    assert top[0]["tool_name"] == "cairn_store"
    assert top[0]["call_count"] == 5
    tracker.close()


def test_local_embedding_zero_cost(tmp_path):
    from cairn.usage_tracker import UsageTracker

    tracker = UsageTracker(db_path=str(tmp_path / "usage.db"))
    tracker.log_call("s1", "embed", "nomic-embed-text", 5000, 0)
    cost = tracker.get_cost_estimate(days=1)
    assert cost["total_usd"] == 0.0
    tracker.close()
