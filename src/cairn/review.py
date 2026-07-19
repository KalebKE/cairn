"""Cairn Review — bridge between Revue code review engine and Cairn memory.

Pro-only module. Uses Revue for multi-agent code review and Cairn's memory
system for persistent codebase context, team conventions, and incident awareness.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Dict, List, Optional

logger = logging.getLogger("cairn.review")


def run_review(
    diff_text: str,
    repo: str = "unknown",
    mode: str = "normal",
    agent_types: Optional[List[str]] = None,
    summarize_only: bool = False,
    session_id: Optional[str] = None,
    entity_id: Optional[str] = None,
) -> str:
    """Run a code review with Cairn memory context.

    Returns a formatted markdown report.
    """
    try:
        from revue.engine import RevueEngine
    except ImportError:
        return (
            "# Review Unavailable\n\n"
            "The `revue` package is not installed. Install it:\n"
            "```\npip install revue\n```"
        )

    # Get Cairn memory context for the review
    conventions = _get_conventions_from_cairn(repo, entity_id)
    patterns = _get_review_patterns_from_cairn(repo, entity_id)

    # Summary-only mode (fast, no LLM)
    if summarize_only:
        engine = RevueEngine(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        summary = engine.summarize(diff_text)
        return _format_summary_report(summary, conventions)

    # Full review with agents
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        # Try Cairn secrets
        try:
            import json
            secrets_path = os.path.expanduser("~/.cairn/secrets.json")
            if os.path.exists(secrets_path):
                with open(secrets_path) as f:
                    secrets = json.load(f)
                api_key = secrets.get("ANTHROPIC_API_KEY")
        except Exception:
            pass

    if not api_key:
        return (
            "# Review Unavailable\n\n"
            "No ANTHROPIC_API_KEY found. Set it in environment or ~/.cairn/secrets.json."
        )

    engine = RevueEngine(api_key=api_key)

    # Inject Cairn conventions into the consistency agent
    if conventions and hasattr(engine.agents.get("consistency", None), "conventions"):
        from revue.storage._types import AgentType
        agent = engine.agents.get(AgentType.CONSISTENCY)
        if agent:
            agent.conventions = conventions

    # Run the review
    result = asyncio.run(engine.review_diff(
        diff_text=diff_text,
        repo=repo,
        agent_types=agent_types,
        mode=mode,
    ))

    # Store findings in Cairn memory for future reference
    if result.findings:
        _store_findings_in_cairn(result, repo, session_id, entity_id)

    return _format_review_report(result, conventions, patterns)


def _get_conventions_from_cairn(repo: str, entity_id: Optional[str] = None) -> List[str]:
    """Query Cairn memory for team conventions relevant to this repo."""
    try:
        from cairn.bridge import query
        results = query(
            query_text=f"coding conventions standards patterns for {repo}",
            limit=10,
            event_type="user_preference",
            entity_id=entity_id,
        )
        if not results:
            return []
        # Parse the query result (it returns formatted text)
        conventions = []
        for line in results.split("\n"):
            line = line.strip()
            if line and not line.startswith("#") and not line.startswith("---"):
                # Extract content from memory entries
                if ":" in line and len(line) > 10:
                    conventions.append(line)
        return conventions[:10]
    except Exception as e:
        logger.debug("Could not fetch Cairn conventions: %s", e)
        return []


def _get_review_patterns_from_cairn(repo: str, entity_id: Optional[str] = None) -> List[Dict]:
    """Query Cairn for past review patterns, incidents, and lessons."""
    try:
        from cairn.bridge import query
        results = query(
            query_text=f"code review findings bugs incidents patterns for {repo}",
            limit=5,
            event_type="lesson_learned",
            entity_id=entity_id,
        )
        if not results:
            return []
        patterns = []
        for line in results.split("\n"):
            line = line.strip()
            if line and len(line) > 20 and not line.startswith("#"):
                patterns.append({"content": line, "source": "cairn_memory"})
        return patterns[:5]
    except Exception:
        return []


def _store_findings_in_cairn(result, repo: str, session_id: Optional[str], entity_id: Optional[str]) -> None:
    """Store significant findings in Cairn memory for future context."""
    try:
        from cairn.bridge import auto_capture

        # Only store critical and major findings
        significant = [f for f in result.findings if f.get("severity") in ("critical", "major")]
        if not significant:
            return

        summary_parts = []
        for f in significant[:5]:
            summary_parts.append(
                f"[{f['severity'].upper()}] {f['agent_type']}: {f['title']} "
                f"({f.get('file_path', 'unknown')})"
            )

        content = (
            f"Code review findings for {repo}:\n"
            + "\n".join(f"- {p}" for p in summary_parts)
        )

        auto_capture(
            content=content,
            event_type="lesson_learned",
            metadata={
                "source": "cairn_review",
                "repo": repo,
                "review_id": result.review_id,
                "finding_count": len(result.findings),
                "critical_count": sum(1 for f in result.findings if f.get("severity") == "critical"),
                "major_count": sum(1 for f in result.findings if f.get("severity") == "major"),
            },
            session_id=session_id,
            entity_id=entity_id,
        )
    except Exception as e:
        logger.debug("Could not store findings in Cairn: %s", e)


def _format_summary_report(summary: Dict, conventions: List[str]) -> str:
    """Format a summary-only report."""
    lines = [
        "# Review Summary",
        "",
        f"**{summary.get('one_liner', 'No changes')}**",
        "",
        f"Risk: **{summary.get('risk', 'low').upper()}** — {summary.get('risk_reason', '')}",
        "",
    ]

    if summary.get("affected_areas"):
        lines.append("**Affected areas:** " + ", ".join(summary["affected_areas"]))
        lines.append("")

    if summary.get("review_focus"):
        lines.append("### Review Focus")
        for focus in summary["review_focus"]:
            lines.append(f"- {focus}")
        lines.append("")

    if conventions:
        lines.append("### Relevant Team Conventions")
        for conv in conventions[:5]:
            lines.append(f"- {conv}")

    return "\n".join(lines)


def _format_review_report(result, conventions: List[str], patterns: List[Dict]) -> str:
    """Format a full review report with Cairn context."""
    summary = result.summary
    findings = result.findings

    lines = [
        "# Code Review Report",
        "",
        f"**{summary.get('files_reviewed', 0)} files reviewed** "
        f"(+{summary.get('lines_added', 0)}/-{summary.get('lines_removed', 0)})",
        "",
    ]

    # Summary stats
    total = summary.get("total_findings", 0)
    filtered = summary.get("filtered_count", 0)
    by_sev = summary.get("by_severity", {})
    lines.append(
        f"**{total} findings** "
        f"({by_sev.get('critical', 0)} critical, {by_sev.get('major', 0)} major, "
        f"{by_sev.get('minor', 0)} minor)"
    )
    if filtered:
        lines.append(f"*{filtered} low-confidence findings filtered*")
    lines.append("")

    if result.review_summary_text:
        lines.append(f"> {result.review_summary_text}")
        lines.append("")

    # Findings by severity
    for severity in ["critical", "major", "minor", "nitpick"]:
        sev_findings = [f for f in findings if f.get("severity") == severity]
        if not sev_findings:
            continue

        lines.append(f"## {severity.upper()} ({len(sev_findings)})")
        lines.append("")

        for f in sev_findings:
            conf = f.get("confidence", 0)
            lines.append(f"### {f['title']}")
            lines.append(f"*{f['agent_type']}* | {conf:.0%} confidence")
            if f.get("file_path"):
                loc = f["file_path"]
                if f.get("line_start"):
                    loc += f":{f['line_start']}"
                lines.append(f"📍 `{loc}`")
            lines.append("")
            lines.append(f["description"])
            if f.get("suggestion"):
                lines.append(f"\n**Fix:** {f['suggestion']}")
            lines.append("")

    # Cairn context section
    if conventions or patterns:
        lines.append("---")
        lines.append("## Cairn Context")
        lines.append("")
        if conventions:
            lines.append("### Team Conventions Applied")
            for conv in conventions[:5]:
                lines.append(f"- {conv}")
            lines.append("")
        if patterns:
            lines.append("### Related Past Findings")
            for p in patterns[:3]:
                lines.append(f"- {p['content'][:120]}")

    return "\n".join(lines)
