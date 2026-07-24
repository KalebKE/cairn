"""Cairn LLM Provider Abstraction.

Thin wrapper over LLM APIs for the optional LLM-assisted features (rollup,
distillation, query expansion). Bring a key from any major provider — set
CAIRN_LLM_PROVIDER to a name below and provide that provider's key (or the
generic CAIRN_LLM_API_KEY).

    CAIRN_LLM_PROVIDER=gemini    GEMINI_API_KEY=...
    CAIRN_LLM_PROVIDER=groq      GROQ_API_KEY=...
    CAIRN_LLM_PROVIDER=deepinfra DEEPINFRA_API_KEY=...

Native SDK providers: anthropic (default), openai. Everything else is
OpenAI-compatible and routed through the openai SDK with a baked-in base_url:
gemini, mistral, deepinfra, groq, together, fireworks, openrouter, xai, ollama
(local, no key). `openai_compat` is the manual escape hatch for anything not
listed — set CAIRN_LLM_BASE_URL + CAIRN_LLM_API_KEY.

Pick a specific model on any provider with CAIRN_LLM_MODEL_FAST /
CAIRN_LLM_MODEL_STANDARD; CAIRN_LLM_BASE_URL overrides the base URL for any
provider. The whole layer is optional — with no key, these features no-op and
Cairn's core memory works unchanged.
"""

import concurrent.futures
import logging
import os
import threading

logger = logging.getLogger("cairn.llm")

# Singleton LLM clients — avoid TCP+TLS handshake per call
_anthropic_client = None
_openai_clients: dict[str, object] = {}  # keyed by (base_url, api_key)
_client_lock = threading.Lock()

# Token usage from the most recent completion, for callers that want it (e.g.
# the provider benchmark). The shared completion path otherwise discards
# response.usage. This only stores a small dict; nothing on the hot path reads
# it, so it adds no cost to normal operation.
_last_usage: dict = {}


def get_last_usage() -> dict:
    """Return {input_tokens, output_tokens, model} from the last completion, or {}."""
    return dict(_last_usage)


def reset_usage() -> None:
    _last_usage.clear()


def reset_clients():
    """Reset all singleton LLM clients. Used by tests."""
    global _anthropic_client
    with _client_lock:
        _anthropic_client = None
        _openai_clients.clear()

# Provider registry. `sdk` selects the client; OpenAI-compatible providers carry
# a base_url. Default model tiers are sensible starting points — override any of
# them with CAIRN_LLM_MODEL_FAST / CAIRN_LLM_MODEL_STANDARD (model IDs move fast).
_PROVIDERS: dict[str, dict] = {
    "anthropic": {"sdk": "anthropic", "key_env": "ANTHROPIC_API_KEY",
                  "fast": "claude-haiku-4-5-20251001", "standard": "claude-sonnet-4-6"},
    "openai": {"sdk": "openai", "key_env": "OPENAI_API_KEY", "base_url": None,
               "fast": "gpt-4o-mini", "standard": "gpt-4o"},
    "gemini": {"sdk": "openai", "key_env": "GEMINI_API_KEY",
               "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
               "fast": "gemini-2.5-flash", "standard": "gemini-2.5-pro"},
    "mistral": {"sdk": "openai", "key_env": "MISTRAL_API_KEY",
                "base_url": "https://api.mistral.ai/v1",
                "fast": "mistral-small-latest", "standard": "mistral-large-latest"},
    "deepinfra": {"sdk": "openai", "key_env": "DEEPINFRA_API_KEY",
                  "base_url": "https://api.deepinfra.com/v1/openai",
                  "fast": "meta-llama/Meta-Llama-3.1-8B-Instruct",
                  "standard": "meta-llama/Llama-3.3-70B-Instruct"},
    "groq": {"sdk": "openai", "key_env": "GROQ_API_KEY",
             "base_url": "https://api.groq.com/openai/v1",
             "fast": "llama-3.1-8b-instant", "standard": "llama-3.3-70b-versatile"},
    "together": {"sdk": "openai", "key_env": "TOGETHER_API_KEY",
                 "base_url": "https://api.together.xyz/v1",
                 "fast": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
                 "standard": "meta-llama/Llama-3.3-70B-Instruct-Turbo"},
    "fireworks": {"sdk": "openai", "key_env": "FIREWORKS_API_KEY",
                  "base_url": "https://api.fireworks.ai/inference/v1",
                  "fast": "accounts/fireworks/models/llama-v3p1-8b-instruct",
                  "standard": "accounts/fireworks/models/llama-v3p3-70b-instruct"},
    "openrouter": {"sdk": "openai", "key_env": "OPENROUTER_API_KEY",
                   "base_url": "https://openrouter.ai/api/v1",
                   "fast": "openai/gpt-4o-mini", "standard": "openai/gpt-4o"},
    "xai": {"sdk": "openai", "key_env": "XAI_API_KEY",
            "base_url": "https://api.x.ai/v1",
            "fast": "grok-3-mini", "standard": "grok-3"},
    "ollama": {"sdk": "openai", "key_env": None,  # local server, no key
               "base_url": "http://localhost:11434/v1",
               "fast": "llama3.2", "standard": "llama3.1:70b"},
    # Manual escape hatch: point at any OpenAI-compatible endpoint.
    "openai_compat": {"sdk": "openai", "key_env": "CAIRN_LLM_API_KEY", "base_url": None,
                      "fast": "default", "standard": "default"},
}


def get_model_map() -> dict[str, dict[str, str]]:
    """Return the provider -> model tier mapping (public API)."""
    return {name: {"fast": s.get("fast", ""), "standard": s.get("standard", "")}
            for name, s in _PROVIDERS.items()}


def list_providers() -> list[str]:
    """Names of all known providers."""
    return list(_PROVIDERS)


def _get_api_key(provider: str) -> str:
    """Resolve the API key for a provider.

    Natives (anthropic/openai) use only their own key env. openai_compat uses
    CAIRN_LLM_API_KEY (default 'none'). The named OpenAI-compatible providers
    use their own key env, then fall back to the generic CAIRN_LLM_API_KEY, so
    a single key var works for any of them. Keyless local servers (ollama) get
    a harmless placeholder.
    """
    if provider == "openai_compat":
        return os.environ.get("CAIRN_LLM_API_KEY", "none")
    spec = _PROVIDERS.get(provider)
    if not spec:
        return ""
    key_env = spec.get("key_env")
    if key_env:
        key = os.environ.get(key_env, "")
        if key:
            return key
    if provider in ("anthropic", "openai"):
        return ""  # natives: no generic fallback (preserves explicit behavior)
    generic = os.environ.get("CAIRN_LLM_API_KEY", "")
    if generic:
        return generic
    return "ollama" if key_env is None else ""


def _get_anthropic_client(api_key: str, timeout: float = 30.0):
    """Get or create a singleton Anthropic client.

    Re-creates the client if the API key changes (e.g., in tests).
    """
    global _anthropic_client
    if _anthropic_client is not None:
        # Check if key changed (test isolation)
        cached_key = getattr(_anthropic_client, "_cairn_api_key", None)
        if cached_key == api_key:
            return _anthropic_client
    with _client_lock:
        if _anthropic_client is not None:
            cached_key = getattr(_anthropic_client, "_cairn_api_key", None)
            if cached_key == api_key:
                return _anthropic_client
        import anthropic

        # Use a generous default timeout for the client; per-call timeouts
        # are enforced via concurrent.futures in llm_complete().
        client = anthropic.Anthropic(api_key=api_key, timeout=timeout)
        client._cairn_api_key = api_key  # Track for invalidation
        _anthropic_client = client
        return _anthropic_client


def _complete_anthropic(
    prompt: str, system: str, *, model: str, max_tokens: int,
    temperature: float, timeout: float,
) -> str:
    """Complete via Anthropic SDK."""
    api_key = _get_api_key("anthropic")
    if not api_key:
        return ""

    client = _get_anthropic_client(api_key, timeout=timeout)
    if client is None:
        return ""

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    try:
        _last_usage.update({
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "model": model,
        })
    except Exception:
        pass
    return response.content[0].text


def _complete_openai(
    prompt: str, system: str, *, model: str, max_tokens: int,
    temperature: float, timeout: float, base_url: str | None = None,
    api_key: str | None = None,
) -> str:
    """Complete via OpenAI SDK (also used for openai_compat)."""
    import openai

    key = api_key or _get_api_key("openai")
    if not key:
        return ""

    # Reuse cached client when possible
    cache_key = (base_url or "", key)
    with _client_lock:
        client = _openai_clients.get(cache_key)
        if client is None:
            kwargs: dict = {"api_key": key, "timeout": max(timeout, 10.0)}
            if base_url:
                kwargs["base_url"] = base_url
            client = openai.OpenAI(**kwargs)
            _openai_clients[cache_key] = client

    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    )
    try:
        _last_usage.update({
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "model": model,
        })
    except Exception:
        pass
    return response.choices[0].message.content or ""


def llm_complete(
    prompt: str,
    system: str,
    *,
    max_tokens: int = 512,
    temperature: float = 0.0,
    timeout: float = 5.0,
    model_tier: str = "fast",
) -> str:
    """Send a prompt to the configured LLM provider. Returns text response.

    Args:
        prompt: User message content.
        system: System prompt.
        max_tokens: Maximum tokens in response.
        temperature: Sampling temperature (0.0 = deterministic).
        timeout: Request timeout in seconds.
        model_tier: "fast" (cheap/quick) or "standard" (capable).

    Returns:
        Response text, or empty string on any failure.
    """
    provider = os.environ.get("CAIRN_LLM_PROVIDER", "anthropic")

    spec = _PROVIDERS.get(provider)
    if not spec:
        logger.warning("Unknown LLM provider: %s (known: %s)", provider, ", ".join(_PROVIDERS))
        return ""

    # Model: registry default for the tier, overridable per tier via env for
    # ANY provider. Resolved at call time (no stale import-snapshot).
    override_env = "CAIRN_LLM_MODEL_STANDARD" if model_tier == "standard" else "CAIRN_LLM_MODEL_FAST"
    model = os.environ.get(override_env) or spec.get(model_tier) or spec.get("fast", "")

    def _do_complete() -> str:
        if spec["sdk"] == "anthropic":
            return _complete_anthropic(
                prompt, system, model=model, max_tokens=max_tokens,
                temperature=temperature, timeout=timeout,
            )
        # All OpenAI-compatible providers: base_url from the registry, but
        # CAIRN_LLM_BASE_URL overrides it (and supplies openai_compat's).
        base_url = os.environ.get("CAIRN_LLM_BASE_URL") or spec.get("base_url")
        return _complete_openai(
            prompt, system, model=model, max_tokens=max_tokens,
            temperature=temperature, timeout=timeout,
            base_url=base_url or None, api_key=_get_api_key(provider),
        )

    # Hard timeout wrapper: ensures we never block longer than timeout,
    # even if the SDK's own timeout handling is slow or broken.
    # NOTE: Don't use `with ThreadPoolExecutor() as ex:` — __exit__ waits
    # for all threads to finish, defeating the timeout.
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(_do_complete)
        return future.result(timeout=timeout + 0.5)
    except concurrent.futures.TimeoutError:
        logger.debug("LLM completion hard timeout (%.1fs) for %s", timeout, provider)
    except Exception as e:
        logger.debug("LLM completion failed (%s): %s", provider, e)
    finally:
        executor.shutdown(wait=False)

    return ""


def claude_complete(
    prompt: str,
    system: str = "You are a helpful assistant providing a second opinion on technical problems.",
    *,
    model: str | None = None,
    max_tokens: int = 4096,
    temperature: float = 0.7,
    timeout: float = 120.0,
) -> str:
    """Consult Claude for a second opinion. Always calls Anthropic regardless of CAIRN_LLM_PROVIDER.

    Returns response text, or empty string on any failure.
    """
    resolved_model = model or os.environ.get("CAIRN_CLAUDE_MODEL", "claude-sonnet-4-6")
    try:
        return _complete_anthropic(
            prompt, system, model=resolved_model, max_tokens=max_tokens,
            temperature=temperature, timeout=timeout,
        )
    except Exception as e:
        logger.debug("Claude consultation failed (%s): %s", resolved_model, e)
        return ""


def gpt_complete(
    prompt: str,
    system: str = "You are a helpful assistant providing a second opinion on technical problems.",
    *,
    model: str | None = None,
    max_tokens: int = 4096,
    temperature: float = 0.7,
    timeout: float = 120.0,
) -> str:
    """Consult GPT for a second opinion. Always calls OpenAI regardless of CAIRN_LLM_PROVIDER.

    Returns response text, or empty string on any failure.
    """
    resolved_model = model or os.environ.get("CAIRN_GPT_MODEL", "gpt-4o")
    try:
        return _complete_openai(
            prompt, system, model=resolved_model, max_tokens=max_tokens,
            temperature=temperature, timeout=timeout,
        )
    except Exception as e:
        logger.debug("GPT consultation failed (%s): %s", resolved_model, e)
        return ""
