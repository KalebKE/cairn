"""
Cairn Types - Constants and event type definitions.
"""

from typing import Dict, Optional


# ============================================================================
# TTL Category Constants for Autonomous Memory Capture
# ============================================================================


class TTLCategory:
    """
    Standardized TTL categories for autonomous memory capture.

    Usage:
        from cairn.types import TTLCategory
        store.store(content, ttl_seconds=TTLCategory.SHORT_TERM)
    """

    EPHEMERAL = 3600  # 1 hour - temporary context, scratch data
    SHORT_TERM = 86400  # 1 day - blocked context, daily work
    LONG_TERM = 7776000  # 90 days - summaries, task completions, git events
    PERMANENT = None  # Never expires - lessons, preferences, error patterns

    @classmethod
    def for_event_type(cls, event_type: str) -> Optional[int]:
        """Get the appropriate TTL for an event type."""
        return EVENT_TYPE_TTL.get(event_type, cls.LONG_TERM)


class AutoCaptureEventType:
    """Standardized event types for autonomous memory capture."""

    # Core events
    SESSION_SUMMARY = "session_summary"
    TASK_COMPLETION = "task_completion"
    ERROR_PATTERN = "error_pattern"
    LESSON_LEARNED = "lesson_learned"
    DECISION = "decision"
    BLOCKED_CONTEXT = "blocked_context"
    USER_PREFERENCE = "user_preference"
    USER_FACT = "user_fact"
    ADVISOR_INSIGHT = "advisor_insight"

    # Git events
    GIT_COMMIT = "git_commit"
    GIT_MERGE = "git_merge"
    GIT_CONFLICT = "git_conflict"

    # Lifecycle events
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    CONTEXT_WARNING = "context_warning"
    BUDGET_ALERT = "budget_alert"

    # Coordination events
    COORDINATION_SNAPSHOT = "coordination_snapshot"

    # Bootstrap events
    PROJECT_CONTEXT = "project_context"

    # Guardrail events
    CONSTRAINT = "constraint"

    # Context virtualization
    CHECKPOINT = "checkpoint"

    # Proactive reminders
    REMINDER = "reminder"

    # Oracle prediction intelligence (pro-only)
    ORACLE_PREDICTION = "oracle_prediction"
    ORACLE_WALLET_SCORE = "oracle_wallet_score"
    ORACLE_REGIME_CHANGE = "oracle_regime_change"
    ORACLE_SIGNAL_SNAPSHOT = "oracle_signal_snapshot"

    # Behavioral pattern inference
    BEHAVIORAL_PATTERN = "behavioral_pattern"

    # Say/Do contradiction tracking (pro-only)
    PUBLIC_STATEMENT = "public_statement"
    OUTCOME_RESOLUTION = "outcome_resolution"
    CONTRADICTION_DETECTED = "contradiction_detected"
    ENTITY_PROFILE_UPDATE = "entity_profile_update"
    PREDICTION_SNAPSHOT = "prediction_snapshot"

    # Experiential memory: distilled session trajectories
    SKILL_TEMPLATE = "skill_template"

    # Cross-session project continuity
    PROJECT_STATUS = "project_status"


# Map event types to TTL categories
EVENT_TYPE_TTL: Dict[str, Optional[int]] = {
    AutoCaptureEventType.SESSION_SUMMARY: TTLCategory.LONG_TERM,  # 90 days: cross-session context carrier
    AutoCaptureEventType.TASK_COMPLETION: TTLCategory.LONG_TERM,
    AutoCaptureEventType.ERROR_PATTERN: TTLCategory.PERMANENT,
    AutoCaptureEventType.LESSON_LEARNED: TTLCategory.PERMANENT,
    AutoCaptureEventType.DECISION: TTLCategory.PERMANENT,  # Architectural knowledge, never expires
    AutoCaptureEventType.BLOCKED_CONTEXT: TTLCategory.SHORT_TERM,
    AutoCaptureEventType.USER_PREFERENCE: TTLCategory.PERMANENT,
    AutoCaptureEventType.ADVISOR_INSIGHT: TTLCategory.PERMANENT,  # System insights = permanent knowledge
    AutoCaptureEventType.GIT_COMMIT: TTLCategory.LONG_TERM,
    AutoCaptureEventType.GIT_MERGE: TTLCategory.LONG_TERM,
    AutoCaptureEventType.GIT_CONFLICT: TTLCategory.PERMANENT,
    AutoCaptureEventType.SESSION_START: TTLCategory.SHORT_TERM,
    AutoCaptureEventType.SESSION_END: TTLCategory.LONG_TERM,
    AutoCaptureEventType.CONTEXT_WARNING: TTLCategory.SHORT_TERM,
    AutoCaptureEventType.BUDGET_ALERT: TTLCategory.LONG_TERM,
    AutoCaptureEventType.COORDINATION_SNAPSHOT: TTLCategory.SHORT_TERM,
    AutoCaptureEventType.PROJECT_CONTEXT: TTLCategory.PERMANENT,
    AutoCaptureEventType.PROJECT_STATUS: TTLCategory.PERMANENT,  # Cross-session project continuity
    "project_history": TTLCategory.PERMANENT,  # LLM rollup syntheses (cairn.rollup) — durable history
    # User-facing types (from legacy/migration)
    "user_fact": TTLCategory.PERMANENT,  # Facts about the user (similar to user_preference)
    "user_prompt": TTLCategory.LONG_TERM,  # Captured user prompts
    "system_event": TTLCategory.SHORT_TERM,  # System-level events
    # Research & evaluation (permanent)
    "sota_research": TTLCategory.PERMANENT,
    "research_report": TTLCategory.PERMANENT,
    "preference_generated": TTLCategory.PERMANENT,
    # Long-term (2 weeks)
    "reflexion": TTLCategory.LONG_TERM,
    "outcome_evaluation": TTLCategory.LONG_TERM,
    "self_reflection": TTLCategory.LONG_TERM,
    "advisor_action_outcome": TTLCategory.LONG_TERM,
    "benchmark_update": TTLCategory.LONG_TERM,
    "file_conflict": TTLCategory.LONG_TERM,
    "session_respawn": TTLCategory.LONG_TERM,
    "memory": TTLCategory.LONG_TERM,  # Generic fallback type for auto_capture; retain 90 days
    # Context virtualization (7 days)
    AutoCaptureEventType.CHECKPOINT: 604800,  # 7 days
    # Proactive reminders (permanent until dismissed)
    AutoCaptureEventType.REMINDER: None,
    # Guardrail constraints (permanent — always enforced)
    AutoCaptureEventType.CONSTRAINT: TTLCategory.PERMANENT,
    # Trajectory distillation (permanent — ACT-R decay handles pruning)
    AutoCaptureEventType.SKILL_TEMPLATE: TTLCategory.PERMANENT,
    # Short-term (1 day)
    "sota_scan": TTLCategory.SHORT_TERM,
    "merge_claim": TTLCategory.SHORT_TERM,
    "merge_release": TTLCategory.SHORT_TERM,
    "file_claimed": TTLCategory.SHORT_TERM,
    "file_released": TTLCategory.SHORT_TERM,
    "branch_claimed": TTLCategory.SHORT_TERM,
    "branch_released": TTLCategory.SHORT_TERM,
    "test": TTLCategory.SHORT_TERM,
    "file_summary": TTLCategory.SHORT_TERM,
    # Ephemeral (1 hour)
    "code_chunk": TTLCategory.EPHEMERAL,
    # Oracle prediction intelligence (pro-only)
    AutoCaptureEventType.ORACLE_PREDICTION: TTLCategory.LONG_TERM,  # 90 days; no query path surfaces these
    AutoCaptureEventType.ORACLE_WALLET_SCORE: TTLCategory.LONG_TERM,
    AutoCaptureEventType.ORACLE_REGIME_CHANGE: TTLCategory.PERMANENT,
    AutoCaptureEventType.ORACLE_SIGNAL_SNAPSHOT: TTLCategory.LONG_TERM,
    # Behavioral pattern inference
    AutoCaptureEventType.BEHAVIORAL_PATTERN: TTLCategory.PERMANENT,
    # Say/Do contradiction tracking (pro-only)
    AutoCaptureEventType.PUBLIC_STATEMENT: TTLCategory.PERMANENT,
    AutoCaptureEventType.OUTCOME_RESOLUTION: TTLCategory.PERMANENT,
    AutoCaptureEventType.CONTRADICTION_DETECTED: TTLCategory.PERMANENT,
    AutoCaptureEventType.ENTITY_PROFILE_UPDATE: TTLCategory.LONG_TERM,
    AutoCaptureEventType.PREDICTION_SNAPSHOT: TTLCategory.LONG_TERM,
}


# ============================================================================
# Stability Classification for Prompt Cache Optimization
# ============================================================================
# Event types whose content rarely changes across sessions.
# Used by hook output builders to place stable content first (cache-friendly prefix)
# and volatile content last (cache-busting suffix).

STABLE_EVENT_TYPES = frozenset({
    "user_preference",
    "user_fact",
    "constraint",
    "decision",
    "lesson_learned",
    "error_pattern",
    "advisor_insight",
    "behavioral_pattern",
    "project_context",
})


__all__ = [
    "TTLCategory",
    "AutoCaptureEventType",
    "EVENT_TYPE_TTL",
    "STABLE_EVENT_TYPES",
]
