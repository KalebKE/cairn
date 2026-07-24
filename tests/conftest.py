"""Cairn test configuration."""
import os
import sys
import pytest
from pathlib import Path

# Ensure cairn package and hooks are importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "hooks"))


@pytest.fixture
def tmp_cairn_dir(tmp_path):
    """Create a temporary Cairn directory for testing."""
    cairn_dir = tmp_path / ".cairn"
    cairn_dir.mkdir()
    os.environ["CAIRN_HOME"] = str(cairn_dir)
    # Default: disable encryption in tests for deterministic output
    old_encrypt = os.environ.get("CAIRN_ENCRYPT")
    os.environ["CAIRN_ENCRYPT"] = "0"
    yield cairn_dir
    os.environ.pop("CAIRN_HOME", None)
    if old_encrypt is not None:
        os.environ["CAIRN_ENCRYPT"] = old_encrypt
    else:
        os.environ.pop("CAIRN_ENCRYPT", None)
    from cairn.crypto import reset_crypto_state
    reset_crypto_state()


@pytest.fixture
def tmp_cairn_dir_encrypted(tmp_path):
    """Create a temporary Cairn directory with encryption enabled."""
    cairn_dir = tmp_path / ".cairn"
    cairn_dir.mkdir()
    os.environ["CAIRN_HOME"] = str(cairn_dir)
    os.environ["CAIRN_ENCRYPT"] = "1"
    from cairn.crypto import reset_crypto_state
    reset_crypto_state()
    yield cairn_dir
    os.environ.pop("CAIRN_HOME", None)
    os.environ.pop("CAIRN_ENCRYPT", None)
    reset_crypto_state()


@pytest.fixture(autouse=True)
def _reset_embeddings_after_test():
    """Reset embedding circuit-breaker after every test to prevent state leaks."""
    yield
    from cairn.embedding import reset_embedding_state
    reset_embedding_state()


@pytest.fixture(autouse=True)
def _hermetic_llm_env(monkeypatch):
    """Clear ambient LLM config so the developer's local hookup (e.g. a
    CAIRN_LLM_PROVIDER=gemini + GOOGLE_API_KEY exported in ~/.zshenv) can't make
    tests hit a real provider. Tests that exercise the LLM set what they need."""
    for v in ("CAIRN_LLM_PROVIDER", "CAIRN_LLM_MODEL_FAST", "CAIRN_LLM_MODEL_STANDARD",
              "CAIRN_LLM_BASE_URL", "CAIRN_LLM_API_KEY", "OPENROUTER_API_KEY",
              "GEMINI_API_KEY", "GOOGLE_API_KEY", "MISTRAL_API_KEY", "ANTHROPIC_API_KEY",
              "OPENAI_API_KEY"):
        monkeypatch.delenv(v, raising=False)


@pytest.fixture
def _reset_bridge(tmp_cairn_dir):
    """Reset the bridge singleton so each test gets a fresh store.

    Centralized definition: test modules can use this via
    @pytest.mark.usefixtures("_reset_bridge") or request it directly.
    Modules that need autouse behavior can define a local autouse fixture
    that depends on this one.
    """
    from cairn.bridge import reset_memory

    reset_memory()
    yield
    reset_memory()


@pytest.fixture
def store(tmp_cairn_dir):
    """Create a fresh SQLiteStore for testing."""
    from cairn.sqlite_store import SQLiteStore
    db_path = tmp_cairn_dir / "test.db"
    s = SQLiteStore(db_path=db_path)
    yield s
    s.close()
