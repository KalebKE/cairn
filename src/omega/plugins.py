"""OMEGA Plugin Interface — extensible discovery for commercial modules.

Core provides entry-point-based plugin discovery. Commercial packages
register via ``[project.entry-points."omega.plugins"]`` in their pyproject.toml.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger("omega.plugins")


class OmegaPlugin:
    """Base class for OMEGA plugins.

    Subclasses should populate these class-level attributes:

    - ``TOOL_SCHEMAS``: list of MCP tool schema dicts
    - ``HANDLERS``: dict mapping tool name → async handler function
    - ``HOOK_HANDLERS``: dict mapping hook name → sync handler function
    - ``CLI_COMMANDS``: list of (name, setup_func) tuples where setup_func(subparsers)
      registers an argparse subparser; the plugin class should also provide a
      ``cmd_{name}(args)`` method as the command handler
    - ``HOOKS_JSON``: dict matching the hooks.json manifest format (optional)
    - ``RETRIEVAL_PROFILES``: dict mapping event_type → (vec, text, word, ctx, graph) phase weights
    - ``SCORE_MODIFIERS``: list of fn(node_id, score, metadata) → score callables
    - ``CAPABILITIES``: set/list of capability strings provided by the plugin,
      for example ``{"unlimited_memory", "full_retrieval", "pro_tools"}``
    """

    TOOL_SCHEMAS: list[dict[str, Any]] = []
    HANDLERS: dict[str, Callable] = {}
    HOOK_HANDLERS: dict[str, Callable] = {}
    CLI_COMMANDS: list[tuple[str, Callable]] = []
    HOOKS_JSON: dict[str, Any] = {}
    RETRIEVAL_PROFILES: dict[str, tuple] = {}
    SCORE_MODIFIERS: list[Callable] = []
    CAPABILITIES: set[str] = set()


def discover_plugins() -> list[OmegaPlugin]:
    """Discover and instantiate all registered OMEGA plugins.

    Looks up ``omega.plugins`` entry-point group via importlib.metadata.
    Each entry point should reference a class that inherits from OmegaPlugin.
    Returns an empty list if no plugins are installed.
    """
    plugins: list[OmegaPlugin] = []
    try:
        from importlib.metadata import entry_points

        eps = entry_points(group="omega.plugins")
        for ep in eps:
            try:
                plugin_cls = ep.load()
                if isinstance(plugin_cls, type) and issubclass(plugin_cls, OmegaPlugin):
                    plugins.append(plugin_cls())
                elif isinstance(plugin_cls, OmegaPlugin):
                    plugins.append(plugin_cls)
                else:
                    logger.warning("Plugin %s is not an OmegaPlugin subclass, skipping", ep.name)
            except Exception as e:
                logger.warning("Failed to load plugin %s: %s", ep.name, e)
    except Exception as e:
        logger.debug("Plugin discovery unavailable: %s", e)
    return plugins


def get_capabilities() -> set[str]:
    """Return capabilities advertised by installed OMEGA plugins.

    Core uses capabilities for feature availability, not payment entitlement.
    A private extension may register capabilities; public Core never trusts a
    local license function to unlock behavior by itself.
    """
    capabilities: set[str] = set()
    for plugin in discover_plugins():
        raw = getattr(plugin, "CAPABILITIES", None)
        if callable(raw):
            raw = raw()
        if not raw:
            raw = getattr(plugin, "capabilities", None)
            if callable(raw):
                raw = raw()
        if not raw:
            continue
        try:
            capabilities.update(str(cap) for cap in raw)
        except TypeError:
            logger.warning("Plugin %s returned invalid capabilities", plugin.__class__.__name__)
    return capabilities


# Fork modification (self-hosted): unlock the capabilities that gate CORE
# data-ownership behavior — full semantic retrieval and the write-count cap —
# so this self-maintained install is not degraded to keyword-only search past
# 2,000 memories or write-capped at 5,000. These flags gate Apache-2.0 core
# logic (retrieval quality in server/handlers.py, the cap in sqlite_store); the
# closed Pro *modules* (coordination/router/knowledge/entity/profile/oracle)
# remain genuinely absent and are NOT unlocked here — ``pro_tools`` stays gated
# so the server does not try to load modules that don't exist.
_FORK_UNLOCKED_CAPABILITIES = frozenset({"full_retrieval", "unlimited_memory"})


def has_capability(name: str) -> bool:
    """Return True when an installed plugin provides ``name``.

    Self-hosted fork: the core data-ownership capabilities in
    ``_FORK_UNLOCKED_CAPABILITIES`` are always available (see note above).
    """
    if name in _FORK_UNLOCKED_CAPABILITIES:
        return True
    return name in get_capabilities()
