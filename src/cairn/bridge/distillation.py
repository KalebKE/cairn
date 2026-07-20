"""Trajectory distillation — peeled from bridge/__init__.py (Wave 4)."""
import logging
from typing import Optional

import cairn.bridge as _bridge
from cairn import json_compat as json

logger = logging.getLogger("cairn.bridge")


# ---------------------------------------------------------------------------
# Public API -- Trajectory Distillation
# ---------------------------------------------------------------------------


def _get_event_type(m) -> str:
    """Extract event_type from a memory (dict or MemoryResult)."""
    if isinstance(m, dict):
        return m.get("event_type", "unknown")
    return getattr(m, "event_type", "unknown")


def _get_content(m) -> str:
    """Extract content from a memory (dict or MemoryResult)."""
    if isinstance(m, dict):
        return m.get("content", "")
    return getattr(m, "content", "") or ""


def _safe_meta(m) -> dict:
    """Extract metadata dict from a memory (dict or MemoryResult)."""
    if isinstance(m, dict):
        meta = m.get("metadata", {})
    else:
        meta = getattr(m, "metadata", {})
    if isinstance(meta, str):
        try:
            return json.loads(meta)
        except Exception:
            return {}
    return meta or {}


def distill_trajectory(session_id: str) -> Optional[str]:
    """Distill a session's memory trajectory into a reusable skill template.

    Called at session stop. Returns the stored node_id, or None if the session
    didn't pass the quality gate or distillation failed.

    Fail-open: any error results in None (no skill stored), never blocks session stop.
    """
    import json as _json

    try:
        db = _bridge._get_store()
        memories = db.get_by_session(session_id, limit=50)

        # Quality gate: minimum 3 memories
        if len(memories) < 3:
            logger.debug("distill_trajectory: skipped session %s (only %d memories)", session_id, len(memories))
            return None

        # Quality gate: must have task_completion event type OR a commit in metadata
        has_completion = any(
            _get_event_type(m) == "task_completion"
            for m in memories
        )
        has_commit = any(
            _safe_meta(m).get("commit")
            for m in memories
        )
        if not has_completion and not has_commit:
            logger.debug("distill_trajectory: skipped session %s (no completion/commit)", session_id)
            return None

        # Gather trajectory context (chronological — oldest first)
        memories = list(reversed(memories))  # get_by_session returns DESC
        mem_lines = []
        for m in memories:
            et = _get_event_type(m)
            content = _get_content(m)[:200]
            mem_lines.append(f"- [{et}] {content}")

        trajectory_text = "\n".join(mem_lines[:20])  # Cap at 20 entries

        # Tool sequence came from a Pro-only audit feature; unavailable here.
        tool_sequence = ""

        # LLM distillation call
        system_prompt = (
            "You extract reusable skill templates from agent work sessions. "
            "Output valid JSON only, no markdown fencing."
        )
        user_prompt = f"""Analyze this agent session and extract a reusable skill template.

Memory sequence (chronological):
{trajectory_text}
{tool_sequence}

Extract a JSON skill template:
{{
  "skill_type": "debugging|feature|refactor|config|deploy",
  "summary": "One sentence describing the workflow in imperative form",
  "steps": ["verb_phrase_1", "verb_phrase_2", ...],
  "key_insight": "The most important actionable lesson from this session",
  "tools_used": ["Tool1", "Tool2"],
  "files_involved": ["path1", "path2"],
  "outcome": "success|partial|failed_then_recovered"
}}

Rules:
- Steps should be abstract enough to transfer (not "edit auth.py line 42" but "apply null-safe fix")
- key_insight should be actionable advice, not a description
- 3-7 steps maximum
- If the session is too routine or trivial to extract a skill, return {{"skip": true}}"""

        raw = _bridge.llm_complete(
            prompt=user_prompt,
            system=system_prompt,
            max_tokens=512,
            temperature=0.0,
            timeout=10.0,
            model_tier="fast",
        )

        if not raw:
            logger.debug("distill_trajectory: LLM returned empty for session %s", session_id)
            return None

        # Parse JSON (strip markdown fencing if present)
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = "\n".join(cleaned.split("\n")[1:])
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("```", 1)[0]
        cleaned = cleaned.strip()

        try:
            skill = _json.loads(cleaned)
        except _json.JSONDecodeError:
            logger.debug("distill_trajectory: malformed JSON from LLM for session %s", session_id)
            return None

        # Handle skip response
        if skill.get("skip"):
            logger.debug("distill_trajectory: LLM said skip for session %s", session_id)
            return None

        # Validate required fields
        required = ("skill_type", "summary", "steps", "key_insight")
        if not all(skill.get(k) for k in required):
            logger.debug("distill_trajectory: missing required fields for session %s", session_id)
            return None

        # Build content string (human-readable)
        steps_str = " → ".join(skill["steps"])
        files_str = ", ".join(skill.get("files_involved", [])[:5])
        content = (
            f"{skill['summary']}. "
            f"Steps: {steps_str}. "
            f"Insight: {skill['key_insight']}"
        )
        if files_str:
            content += f". Files: {files_str}"

        # Build metadata
        meta = {
            "source": "trajectory_distillation",
            "session_id": session_id,
            "skill_type": skill["skill_type"],
            "steps": skill["steps"],
            "tools_used": skill.get("tools_used", []),
            "files_involved": skill.get("files_involved", []),
            "key_insight": skill["key_insight"],
            "outcome": skill.get("outcome", "success"),
            "memory_count": len(memories),
            "distillation_model": "haiku",
        }

        node_id = _bridge.auto_capture(
            content=content,
            event_type="skill_template",
            metadata=meta,
            session_id=session_id,
        )

        logger.info("distill_trajectory: distilled %s skill from session %s → %s",
                     skill["skill_type"], session_id, node_id)
        return node_id

    except Exception as e:
        logger.debug("distill_trajectory: failed for session %s: %s", session_id, e)
        return None


# ---------------------------------------------------------------------------
