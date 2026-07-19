# LLM Routing

## Overview

The Cairn router classifies prompts by intent and routes them to the optimal LLM based on priority mode, context affinity, and token budget. The classifier runs locally via ONNX prototypes in under 2 milliseconds --- no external API calls for classification.

Install: `pip install cairn[router]`

The router supports 5 intents, 5 providers, and 4 priority modes. It tracks which model you are currently using (context affinity) to avoid unnecessary switches, and automatically routes to large-context models (Gemini) when prompts exceed 100K tokens.

## Quick Example

```
# Classify a prompt
cairn_classify_intent(prompt="Write a recursive tree traversal in Python")
# Returns: intent="coding", confidence=0.92

# Route to the best model
cairn_route_prompt(prompt="Write a recursive tree traversal in Python", priority="quality")
# Returns: recommended model, provider, and reasoning

# Switch priority mode for the session
cairn_set_priority_mode(mode="speed")
```

## Intents

| Intent | Description | Example Prompts |
|--------|-------------|-----------------|
| `coding` | Writing, debugging, or reviewing code | "Fix the null pointer in auth.py", "Write a REST endpoint" |
| `creative` | Writing prose, brainstorming, content generation | "Draft a blog post about microservices", "Name this feature" |
| `logic` | Reasoning, math, analysis, architecture decisions | "Compare PostgreSQL vs DynamoDB for this use case" |
| `exploration` | Research, open-ended questions, broad investigation | "How does Kubernetes handle pod scheduling?" |
| `simple_edit` | Small edits, renames, formatting, typo fixes | "Rename this variable", "Add a docstring to this function" |

## Providers and Models

| Provider | Models | Strengths |
|----------|--------|-----------|
| Anthropic | Claude Opus 4.6, Claude Sonnet | Coding, reasoning |
| OpenAI | GPT-4o, GPT-4o-mini | General purpose, creative |
| Google | Gemini 2.5 Pro/Flash, Gemini 3 Pro/Flash | Large context (1M+ tokens) |
| Groq | Llama 3.1 8B Instant | Speed, simple edits |
| xAI | Grok-4, Grok-4.1-fast | Exploration, long context (2M) |

## Priority Modes

| Mode | Behavior |
|------|----------|
| `cost` | Route to the cheapest model that can handle the intent |
| `speed` | Route to the fastest model (Groq for simple edits, smaller models elsewhere) |
| `quality` | Route to the most capable model for the intent |
| `balanced` | Default. Weighs cost, speed, and quality equally. |

## Tools Reference

| Tool | Purpose |
|------|---------|
| `cairn_route_prompt` | Route a prompt to the optimal LLM. Accepts priority override and estimated token count. |
| `cairn_classify_intent` | Classify a prompt's intent without routing. Returns intent, confidence, and all scores. |
| `cairn_router_status` | Show provider availability (which have API keys), routing stats, and current priority mode |
| `cairn_set_priority_mode` | Set the routing priority mode for all subsequent routing decisions |
| `cairn_get_model_config` | View routing configuration for a specific intent or all intents |
| `cairn_switch_model` | Switch to a different LLM with Cairn memory preservation. Retrieves relevant context for the target model. |
| `cairn_get_current_model` | Get the current model for a session (from context affinity tracking) |
| `cairn_router_context` | Get session context: current model, provider, token count, conversation depth |
| `cairn_warm_router` | Pre-load the intent classifier prototypes and model config. Reduces first-route latency. |
| `cairn_router_benchmark` | Run a quick accuracy test: 6 sample prompts, verify intent classification and routing |

## Setup

Store API keys in `~/.cairn/secrets.json` (file is chmod 600):

```json
{
  "anthropic_api_key": "<anthropic-api-key>",
  "openai_api_key": "<openai-api-key>",
  "google_api_key": "<google-api-key>",
  "groq_api_key": "<groq-api-key>",
  "xai_api_key": "<xai-api-key>"
}
```

You can also set keys via environment variables in `~/.zshrc`:
```bash
export ANTHROPIC_API_KEY="<anthropic-api-key>"
export OPENAI_API_KEY="<openai-api-key>"
```

Verify setup:
```
cairn_router_status
# Shows which providers have valid keys
```

## Common Workflows

### Basic Routing

Set a priority mode once, then route prompts:

```
cairn_set_priority_mode(mode="balanced")

cairn_route_prompt(prompt="Implement a rate limiter with sliding window")
# Returns: Anthropic/Claude Opus 4.6 (coding intent, quality match)

cairn_route_prompt(prompt="Fix the typo in the README")
# Returns: Groq/Llama 3.1 8B (simple_edit intent, speed match)
```

Override priority per-prompt:
```
cairn_route_prompt(prompt="Analyze this architecture", priority="quality")
```

### Intent Classification

Classify without routing --- useful for understanding how the router sees your prompts:

```
cairn_classify_intent(prompt="Should we use a message queue or direct HTTP calls?", detailed=True)
# Returns:
#   intent: logic
#   confidence: 0.87
#   scores: {logic: 0.87, coding: 0.45, exploration: 0.38, creative: 0.12, simple_edit: 0.03}
```

### Large Context Override

When a prompt involves more than 100K tokens (e.g., analyzing a large codebase), the router automatically overrides to a large-context model:

```
cairn_route_prompt(prompt="Analyze this codebase for security issues", estimated_tokens=150000)
# Returns: Google/Gemini or xAI/Grok-4.1-fast (large context capable)
```

### Context Affinity

The router tracks which model you are currently using and penalizes unnecessary switches. This prevents thrashing between providers mid-conversation:

```
cairn_router_context(session_id="agent-1")
# Returns: current model, provider, token count, conversation depth
```

### Model Switching

When you need to switch providers (e.g., moving from coding to creative work), use `cairn_switch_model` to preserve Cairn memory context:

```
cairn_switch_model(
    session_id="agent-1",
    target_provider="openai",
    target_model="gpt-4o",
    retrieve_context=True
)
# Retrieves relevant Cairn context and adapts it for the target model
```

### Warm-Up

Pre-load the classifier at session start to eliminate first-route latency:

```
cairn_warm_router()
```

### Benchmarking

Verify the router is working correctly:

```
cairn_router_benchmark()
# Runs 6 sample prompts, reports intent classification accuracy and routing decisions
```

### Cross-Model Consultation

Cairn includes two consultation tools that let you get a second opinion from a different LLM provider. The system is provider-aware: if you are running on Claude, you get `cairn_consult_gpt`. If you are running on OpenAI or another non-Anthropic provider, you get `cairn_consult_claude`.

**When to use consultation:**
- Stuck on a problem for 10+ minutes or after 3+ failed approaches
- Facing an irreversible architecture decision
- Debugging a dead end
- Cross-validating a fragile solution

**Example (from a Claude agent):**
```
cairn_consult_gpt(
    prompt="Should I use a message queue or direct HTTP calls for this microservice?",
    context="We have 3 services, ~100 req/sec, need exactly-once delivery for payment events",
    temperature=0.5
)
# Returns: GPT's analysis with a different perspective
```

**Example (from an OpenAI agent):**
```
cairn_consult_claude(
    prompt="Review this SQL migration for edge cases",
    context="ALTER TABLE users ADD COLUMN email_verified BOOLEAN DEFAULT FALSE; ...",
    temperature=0.3
)
```

Consultation requires the target provider's API key in `~/.cairn/secrets.json`. If the key is missing, the tool is unavailable (no error, just not shown in the tool list).

## Tips

- **Warm the router at session start.** Calling `cairn_warm_router` loads ONNX prototypes into memory, reducing first-route latency from ~50ms to <2ms.
- **Classification is free.** `cairn_classify_intent` runs entirely locally via ONNX --- no API calls, no cost, sub-2ms latency. Use it liberally.
- **Priority modes are session-wide.** `cairn_set_priority_mode` affects all subsequent routing in the session. Override per-prompt with the `priority` parameter on `cairn_route_prompt`.
- **Context affinity reduces churn.** The router prefers to stay on the current model. It will only switch when a different model is significantly better for the intent.
- **Estimated tokens matter.** If you know a prompt involves a large context, pass `estimated_tokens` to trigger the large-context override. Without it, the router cannot know the full context size.
- **Secrets file permissions.** `~/.cairn/secrets.json` should be chmod 600. The CLI `cairn setup` sets this automatically.
- **Not all providers are required.** The router works with whatever providers have valid keys. Missing providers are simply skipped during routing.
- **Force an intent.** If you disagree with the classifier, use `force_intent` on `cairn_route_prompt` to override classification.
