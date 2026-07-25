"""Shared task text utilities for hooks and status bar."""
import logging
import re

logger = logging.getLogger(__name__)

# System-generated prefixes that aren't real tasks
_SKIP_PREFIXES = (
    "MEMORY HANDOFF",
    "[Request interrupted",
    "Implement the following plan",
    "Execute this plan",
    "Follow this plan",
    "Here is the plan",
)

# Short non-task messages
_SKIP_EXACT = frozenset({
    "proceed", "continue", "yes", "no", "ok", "go ahead", "lgtm",
})


def clean_task_text(prompt: str) -> str:
    """Extract a clean, short task description from a user prompt.

    Strips XML tags, system prefixes, markdown headers.  Splits at the first
    sentence boundary (letter followed by `. `) and caps at 60 chars.

    Returns empty string if the prompt isn't a real task.
    """
    text = prompt.strip()
    if not text:
        return ""

    # Strip XML-like tags
    text = re.sub(r"<[a-zA-Z_-]+>[^<]*</[a-zA-Z_-]+>", "", text)
    text = re.sub(r"</?[a-zA-Z_-]+>", "", text)
    text = text.strip()

    # Skip system-generated / non-task messages
    for pfx in _SKIP_PREFIXES:
        if text.startswith(pfx):
            return ""
    if text.lower() in _SKIP_EXACT:
        return ""

    # Strip markdown heading markers
    text = re.sub(r"^#{1,4}\s*", "", text)
    # Strip "Resume:" prefix
    text = re.sub(r"^[Rr]esume\s*:\s*", "", text)

    # First line only (multi-line prompts: first line is usually the task)
    text = text.split("\n")[0].strip()

    # Sentence split: only at letter + punctuation + space (not "v2." or "87.")
    text = re.sub(r"([a-zA-Z])[.!?]\s.*", r"\1", text)
    text = text.strip()

    # Cap at 60 chars, cut at last space if mid-word
    if len(text) > 60:
        text = text[:60]
        if " " in text:
            text = text[:text.rfind(" ")]

    return text if len(text) >= 8 else ""


# summarize_task_text (LLM-backed title summarizer) and its _basic_clean
# feeder were removed 2026-07-25: no callers anywhere in the repo or in the
# installed hooks, and removing them keeps this module free of LLM imports —
# all remote-LLM usage lives on the async maintenance paths.
