"""Cairn integration for LangChain / LangGraph agents.

Provides a simple interface for LangChain agents to store and retrieve
memories via Cairn's local semantic search engine.

Usage::

    from cairn.integrations.langchain import CairnMemory

    mem = CairnMemory()

    # Store context from a chain run
    mem.save("User prefers PostgreSQL over MongoDB for ACID transactions")

    # Retrieve relevant memories for a query
    results = mem.recall("What database should I use for the orders service?")
    # Returns: ["User prefers PostgreSQL over MongoDB for ACID transactions"]

    # Use as context injection in a LangChain chain
    from langchain_core.prompts import ChatPromptTemplate

    context = mem.recall_as_context("database choice")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Context from memory:\\n{memory}"),
        ("human", "{input}"),
    ])
    chain = prompt | llm
    chain.invoke({"input": "What DB should I use?", "memory": context})

Requires: ``pip install cairn langchain-core``
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class CairnMemory:
    """Simple Cairn memory interface for LangChain/LangGraph agents.

    Unlike the deprecated BaseMemory, this is a standalone helper that
    can be composed into any chain or graph node.
    """

    def __init__(self, *, project: str = "langchain", user_id: str = "default"):
        from cairn.bridge import (
            query as cairn_query,
            store as cairn_store,
        )
        self._store_fn = cairn_store
        self._query_fn = cairn_query
        self._project = project
        self._user_id = user_id

    def save(
        self,
        content: str,
        event_type: str = "memory",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Store a memory in Cairn.

        Args:
            content: The text to remember.
            event_type: One of: memory, decision, lesson_learned, error_pattern,
                user_preference, task_completion.
            metadata: Optional metadata dict.

        Returns:
            Confirmation message.
        """
        meta = metadata or {}
        meta["source"] = f"langchain:{self._user_id}"
        return self._store_fn(
            content=content,
            event_type=event_type,
            metadata=meta,
            project=self._project,
        )

    def save_context(
        self,
        inputs: dict[str, Any],
        outputs: dict[str, Any],
    ) -> None:
        """Save the input/output pair from a chain run.

        Compatible with the LangChain memory callback pattern.
        """
        input_str = " ".join(str(v) for v in inputs.values())
        output_str = " ".join(str(v) for v in outputs.values())
        content = f"Input: {input_str}\nOutput: {output_str}"
        self.save(content, event_type="memory")

    def recall(
        self,
        query: str,
        limit: int = 5,
        event_type: str | None = None,
    ) -> list[str]:
        """Retrieve relevant memories.

        Args:
            query: Natural language query.
            limit: Maximum results.
            event_type: Filter by type.

        Returns:
            List of memory content strings, ranked by relevance.
        """
        result = self._query_fn(
            query=query,
            limit=limit,
            event_type=event_type,
            project=self._project,
        )
        # Parse Cairn's markdown response to extract memory contents
        memories = []
        if isinstance(result, str):
            for line in result.split("\n"):
                line = line.strip()
                if line.startswith("**Content:**"):
                    memories.append(line.replace("**Content:**", "").strip())
                elif line and not line.startswith(("#", "**", "---", "Results:", "No ")):
                    if len(line) > 10:  # skip short formatting lines
                        memories.append(line)
        return memories[:limit]

    def recall_as_context(self, query: str, limit: int = 5) -> str:
        """Retrieve memories formatted as a context string for prompts.

        Args:
            query: Natural language query.
            limit: Maximum memories to include.

        Returns:
            Newline-separated string of relevant memories, or
            "No relevant memories found." if empty.
        """
        memories = self.recall(query, limit=limit)
        if not memories:
            return "No relevant memories found."
        return "\n".join(f"- {m}" for m in memories)

    def clear(self) -> None:
        """Clear all memories for this project. Use with caution."""
        logger.warning("CairnMemory.clear() called -- this is a no-op to prevent data loss. "
                       "Use Cairn CLI to manage memories: cairn consolidate")
