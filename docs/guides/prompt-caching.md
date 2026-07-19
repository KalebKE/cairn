# Prompt Caching with Cairn Context

## The Pattern

When you build an agent that uses Cairn for memory, a common pattern emerges: every API call sends the same growing prefix (system prompt + Cairn context + conversation history) with only the latest user message changing.

Without caching, you pay full input token pricing on every call. With **automatic prompt caching**, the repeated prefix is cached after the first call, and subsequent calls pay only **10% of the base input price** for the cached portion.

## Automatic Caching (Recommended)

Add a single `cache_control` field at the top level of your request. The API automatically caches everything up to the last cacheable block and moves the breakpoint forward as conversations grow.

```python
import anthropic

client = anthropic.Anthropic()

# Fetch Cairn context once at the start of a conversation
cairn_context = get_cairn_context()  # your function to query Cairn

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=4096,
    cache_control={"type": "ephemeral"},  # one line enables auto-caching
    system=f"""You are a coding assistant with persistent memory.

## Prior Context from Cairn
{cairn_context}
""",
    messages=conversation_history,
)
```

On the first call, the system prompt + Cairn context is written to cache. On every subsequent call in that conversation, it's read from cache at 10% cost.

## Multi-Turn Conversations

Automatic caching handles multi-turn conversations without any extra work. The cache breakpoint moves forward automatically:

| Turn | What's Cached | What's New |
|------|---------------|------------|
| Turn 1 | System + Cairn context + User:A written to cache | Everything is a cache write |
| Turn 2 | System + Cairn context + User:A read from cache | Asst:B + User:C written to cache |
| Turn 3 | System through User:C read from cache | Asst:D + User:E written to cache |

## Agentic Tool Use

If your agent uses tools (MCP or otherwise), each tool call round-trip is a new API request with the full conversation resent. This is where caching saves the most: the growing conversation is cached, and each tool call only pays 10% for the prefix.

```python
# Agent loop with tool use
while True:
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        cache_control={"type": "ephemeral"},
        system=system_with_cairn_context,
        tools=tool_definitions,
        messages=conversation,
    )

    if response.stop_reason == "tool_use":
        # Execute tool, append result, loop back
        # The entire conversation up to the previous turn is a cache hit
        conversation.append({"role": "assistant", "content": response.content})
        conversation.append({"role": "user", "content": tool_results})
    else:
        break
```

## Explicit Breakpoints for Mixed-Frequency Content

If your Cairn context changes more frequently than your system prompt or tool definitions, use explicit breakpoints to cache them independently:

```python
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=4096,
    cache_control={"type": "ephemeral"},  # auto-cache conversation
    system=[
        {
            "type": "text",
            "text": "You are a coding assistant with persistent memory.",
            "cache_control": {"type": "ephemeral"},  # rarely changes
        },
        {
            "type": "text",
            "text": f"## Cairn Context\n{cairn_context}",
            "cache_control": {"type": "ephemeral"},  # changes per session
        },
    ],
    messages=conversation,
)
```

This way, if you refresh the Cairn context mid-conversation, the system prompt cache is preserved while only the Cairn block is rewritten.

## Pricing

Cache reads cost 10% of base input price. Cache writes cost 125% of base. The break-even point is **2 cache reads per write**, meaning caching pays for itself after just 2 turns in a conversation.

| Model | Base Input | Cache Write | Cache Read | Savings at 10 turns |
|-------|-----------|-------------|------------|---------------------|
| Sonnet 4.6 | $3/MTok | $3.75/MTok | $0.30/MTok | ~87% on cached prefix |
| Haiku 4.5 | $1/MTok | $1.25/MTok | $0.10/MTok | ~87% on cached prefix |
| Opus 4.6 | $5/MTok | $6.25/MTok | $0.50/MTok | ~87% on cached prefix |

## Requirements

- **Minimum cacheable tokens**: 1024 (Sonnet/Opus 4.x), 4096 (Opus 4.5+, Haiku 4.5)
- **Cache lifetime**: 5 minutes (default), refreshed on each hit. Optional 1-hour TTL at 2x base cost.
- **Max breakpoints**: 4 per request (automatic caching uses 1 slot)
- **Prefix must be identical**: Any change to cached content invalidates that cache segment and everything after it

## When to Use the 1-Hour TTL

The default 5-minute cache works for active conversations. Use the 1-hour TTL when:

- Your agent has long-running sub-tasks (>5 min between API calls)
- Users may pause and resume conversations
- You're running batch evaluations with the same Cairn context

```python
cache_control={"type": "ephemeral", "ttl": "1h"}
```

## Cairn-Specific Tips

1. **Front-load Cairn context**: Place it in the system prompt, not in user messages. System prompts are cached first in the hierarchy (`tools` > `system` > `messages`).

2. **Stable context ordering**: Cairn query results should be deterministically ordered. If the order changes between calls, the cache is invalidated.

3. **Batch Cairn queries**: Make one `cairn_query` call with broad context rather than many narrow calls. A single large cached block is more efficient than many small ones.

4. **Session-scoped context**: Fetch Cairn context once at session start and inject it into the system prompt. Avoid re-querying Cairn on every turn unless the context genuinely needs refreshing.
