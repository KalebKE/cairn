"""Cairn LLM Provider Abstraction.

Thin wrapper over LLM APIs for text completion. Supports swappable
providers via CAIRN_LLM_PROVIDER env var.

Providers:
  - anthropic (default): Uses anthropic SDK
  - openai: Uses openai SDK
  - openai_compat: Uses openai SDK with custom base_url (for vLLM, MiniMax, etc.)
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


def reset_clients():
    """Reset all singleton LLM clients. Used by tests."""
    global _anthropic_client
    with _client_lock:
        _anthropic_client = None
        _openai_clients.clear()

# Model tier -> provider-specific model name
_MODEL_MAP: dict[str, dict[str, str]] = {
    "anthropic": {
        "fast": "claude-haiku-4-5-20251001",
        "standard": "claude-sonnet-4-6",
    },
    "openai": {
        "fast": os.environ.get("CAIRN_LLM_MODEL_FAST", "gpt-4o-mini"),
        "standard": os.environ.get("CAIRN_LLM_MODEL_STANDARD", "gpt-4o"),
    },
    "openai_compat": {
        "fast": os.environ.get("CAIRN_LLM_MODEL_FAST", "default"),
        "standard": os.environ.get("CAIRN_LLM_MODEL_STANDARD", "default"),
    },
}


def get_model_map() -> dict[str, dict[str, str]]:
    """Return the provider -> model tier mapping (public API)."""
    return _MODEL_MAP


def _get_api_key(provider: str) -> str:
    """Resolve API key for the given provider."""
    if provider == "anthropic":
        return os.environ.get("ANTHROPIC_API_KEY", "")
    if provider == "openai":
        return os.environ.get("OPENAI_API_KEY", "")
    if provider == "openai_compat":
        return os.environ.get("CAIRN_LLM_API_KEY", "none")
    return ""


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

    models = _MODEL_MAP.get(provider)
    if not models:
        logger.warning("Unknown LLM provider: %s", provider)
        return ""

    model = models.get(model_tier, models["fast"])
    if provider == "openai_compat":
        # Resolve at call time: _MODEL_MAP snapshots env at import, which goes
        # stale when CAIRN_LLM_MODEL_* is set after cairn.llm is imported.
        _env_key = f"CAIRN_LLM_MODEL_{'STANDARD' if model_tier == 'standard' else 'FAST'}"
        model = os.environ.get(_env_key, model)

    def _do_complete() -> str:
        if provider == "anthropic":
            return _complete_anthropic(
                prompt, system, model=model, max_tokens=max_tokens,
                temperature=temperature, timeout=timeout,
            )
        if provider == "openai":
            return _complete_openai(
                prompt, system, model=model, max_tokens=max_tokens,
                temperature=temperature, timeout=timeout,
            )
        if provider == "openai_compat":
            base_url = os.environ.get("CAIRN_LLM_BASE_URL", "")
            compat_key = _get_api_key("openai_compat")
            return _complete_openai(
                prompt, system, model=model, max_tokens=max_tokens,
                temperature=temperature, timeout=timeout,
                base_url=base_url or None, api_key=compat_key,
            )
        return ""

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
