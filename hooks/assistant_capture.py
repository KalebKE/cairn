#!/usr/bin/env python3
"""Cairn Stop hook fallback — capture high-value assistant responses.

Fires on every Stop event when the hook daemon is unavailable.
Detects fix, decision, lesson, and recommendation patterns in
``last_assistant_message`` and stores them via bridge.auto_capture.

This is the cold-path fallback. The fast path routes through the hook
daemon to ``handle_assistant_capture`` in the hook_server package.
"""
import json
import re
import sys

# Patterns (mirrors assistant.py in hook_server)
FIX_PATTERNS = [
    r"the (?:fix|issue|problem|bug) was\b",
    r"root cause (?:was|is)\b",
    r"the error (?:occurred|happens|was caused) because\b",
    r"fixed (?:by|this by)\b",
]

DECISION_PATTERNS = [
    r"(?:decided|choosing) to\b",
    r"going with\b",
    r"switched to\b",
    r"using \S+ instead of\b",
    r"chose \S+ because\b",
]

LESSON_PATTERNS = [
    r"(?:note|notice) that\b",
    r"important:\s",
    r"be careful\b",
    r"gotcha:\s",
    r"caveat:\s",
    r"key takeaway\b",
]

MIN_MESSAGE_LENGTH = 200
MIN_CONTENT_CHARS = 40
MIN_CONTENT_WORDS = 8

_FENCED_CODE_RE = re.compile(r"```[\s\S]*?```", re.DOTALL)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_INSIGHT_OPEN_RE = re.compile(r"[★✦]\s*Insight\s*─+", re.IGNORECASE)
_INSIGHT_CLOSE_RE = re.compile(r"─{10,}")

_captured_count = 0
MAX_CAPTURES = 10


def _clean(text):
    return _FENCED_CODE_RE.sub("", text).strip()


def _find_match(text, patterns):
    for pat in patterns:
        compiled = re.compile(pat, re.IGNORECASE)
        for sentence in _SENTENCE_SPLIT_RE.split(text):
            sentence = sentence.strip()
            if compiled.search(sentence):
                if len(sentence) >= MIN_CONTENT_CHARS and len(sentence.split()) >= MIN_CONTENT_WORDS:
                    return sentence
    return None


def _extract_insight_blocks(text):
    """Extract ★ Insight delimited blocks from assistant text."""
    blocks = []
    search_start = 0
    while True:
        open_match = _INSIGHT_OPEN_RE.search(text, search_start)
        if not open_match:
            break
        body_start = open_match.end()
        close_match = _INSIGHT_CLOSE_RE.search(text, body_start)
        if not close_match:
            body = text[body_start:body_start + 2000].strip()
        else:
            body = text[body_start:close_match.start()].strip()
        if body and len(body) >= MIN_CONTENT_CHARS:
            blocks.append(body[:2000])
        search_start = close_match.end() if close_match else len(text)
    return blocks


def main(data=None):
    global _captured_count
    if _captured_count >= MAX_CAPTURES:
        return

    if data is None:
        try:
            raw = sys.stdin.read()
            if not raw.strip():
                return
            data = json.loads(raw)
        except (json.JSONDecodeError, Exception):
            return

    message = data.get("last_assistant_message", "")
    if not message or len(message) < MIN_MESSAGE_LENGTH:
        return

    session_id = data.get("session_id", "")
    cwd = data.get("cwd", data.get("project", ""))

    # Pre-pass: detect ★ Insight delimited blocks
    insight_blocks = _extract_insight_blocks(message)
    if insight_blocks and _captured_count < MAX_CAPTURES:
        for block in insight_blocks:
            if _captured_count >= MAX_CAPTURES:
                break
            try:
                from cairn.bridge import auto_capture

                auto_capture(
                    content=f"Insight: {block}",
                    event_type="advisor_insight",
                    metadata={"source": "assistant_capture_hook", "project": cwd, "capture_confidence": "high", "category": "system_insight"},
                    session_id=session_id,
                    project=cwd,
                )
                _captured_count += 1
                preview = block[:80].replace("\n", " ").strip()
                print(f"[LEARNED] insight: {preview}")
            except ImportError:
                pass
            except Exception:
                pass
        return

    cleaned = _clean(message)
    if not cleaned:
        return

    # Try pattern groups in priority order
    for label, event_type, patterns in [
        ("fix", "lesson_learned", FIX_PATTERNS),
        ("decision", "decision", DECISION_PATTERNS),
        ("lesson", "lesson_learned", LESSON_PATTERNS),
    ]:
        content = _find_match(cleaned, patterns)
        if content:
            try:
                from cairn.bridge import auto_capture

                auto_capture(
                    content=f"Assistant {label}: {content[:500]}",
                    event_type=event_type,
                    metadata={"source": "assistant_capture_hook", "project": cwd},
                    session_id=session_id,
                    project=cwd,
                )
                _captured_count += 1
                preview = content[:80].replace("\n", " ").strip()
                print(f"[LEARNED] {label}: {preview}")
            except ImportError:
                pass
            except Exception:
                pass
            return


if __name__ == "__main__":
    main()
