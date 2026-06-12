"""Tests for aggregate context packet telemetry."""


def test_track_context_packet_aggregates_without_content(tmp_omega_dir, monkeypatch):
    from omega import telemetry

    monkeypatch.setattr(telemetry, "OMEGA_DIR", tmp_omega_dir)
    monkeypatch.setattr(telemetry, "TELEMETRY_FILE", tmp_omega_dir / "telemetry.json")

    telemetry.track_context_packet({
        "mode": "before_edit",
        "memories_used": 3,
        "chain_count": 2,
        "warnings_count": 1,
        "estimated_tokens": 420,
        "estimated_tokens_saved": 1200,
    }, surface="hook")
    telemetry.track_context_packet({
        "mode": "planning",
        "memories_used": 0,
        "chain_count": 0,
        "warnings_count": 0,
        "estimated_tokens": 40,
        "estimated_tokens_saved": 0,
    }, surface="mcp")

    summary = telemetry.get_summary()
    packets = summary["context_packets"]
    assert packets["total"] == 2
    assert packets["with_memories"] == 1
    assert packets["with_chains"] == 1
    assert packets["warnings"] == 1
    assert packets["tokens"] == 460
    assert packets["tokens_saved"] == 1200
    assert packets["by_mode"] == {"before_edit": 1, "planning": 1}
    assert packets["by_surface"] == {"hook": 1, "mcp": 1}

    raw = (tmp_omega_dir / "telemetry.json").read_text()
    assert "local-first" not in raw
    assert "mem-" not in raw
