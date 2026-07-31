"""Vector search must survive an extension-less SQLite.

GitHub's macOS runners shipped a CPython without loadable-extension support
and vector retrieval silently died (the ``top-0 was: []`` incident). These
tests pin the contract: a store built without sqlite-vec still stores
embeddings, still serves semantic queries and find_similar, and a store
built WITH sqlite-vec keeps its embeddings readable if the extension later
disappears (the side table is canonical; vec0 is an index).
"""

import builtins

import pytest

from cairn.sqlite_store import SQLiteStore


@pytest.fixture
def no_vec(monkeypatch):
    """Make ``import sqlite_vec`` fail, reproducing extension-less SQLite."""
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "sqlite_vec":
            raise ImportError("sqlite_vec blocked (extension-less repro)")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)


def _mk_store(tmp_path, name="novec.db"):
    return SQLiteStore(db_path=str(tmp_path / name))


SEMANTIC_MEMORY = (
    "I grow tomatoes, basil and zucchini in my backyard garden and prefer "
    "cooking dinner from whatever is ripe."
)
DISTRACTORS = [
    "Deployment pipeline note {i}: CI stage ordering, artifact caching and "
    "rollback procedure for service {i}.",
    "Kotlin build cache invalidation rules for module {i} on the CI runner.",
]


class TestNoVecStore:
    def test_semantic_query_works_without_sqlite_vec(self, tmp_path, no_vec):
        store = _mk_store(tmp_path)
        try:
            assert store._vec_available is False
            nid = store.store(
                content=SEMANTIC_MEMORY, metadata={"event_type": "user_preference"}
            )
            for i in range(6):
                store.store(
                    content=DISTRACTORS[i % 2].format(i=i),
                    metadata={"event_type": "memory"},
                )
            results = store.query(
                "what vegetables do I have for cooking at home?",
                limit=5,
                use_cache=False,
            )
            assert any(r.id == nid for r in results), (
                f"semantic retrieval dead without sqlite-vec; got "
                f"{[(r.id, (r.content or '')[:30]) for r in results]}"
            )
        finally:
            store.close()

    def test_find_similar_works_without_sqlite_vec(self, tmp_path, no_vec):
        from cairn.embedding import generate_embedding

        store = _mk_store(tmp_path)
        try:
            nid = store.store(content=SEMANTIC_MEMORY, metadata={"event_type": "memory"})
            store.store(
                content=DISTRACTORS[0].format(i=1), metadata={"event_type": "memory"}
            )
            emb = generate_embedding("homegrown vegetables for dinner", mode="query")
            results = store.find_similar(emb, limit=3)
            assert any(r.id == nid for r in results)
        finally:
            store.close()

    def test_deleting_memory_removes_stored_embedding(self, tmp_path, no_vec):
        store = _mk_store(tmp_path)
        try:
            nid = store.store(content=SEMANTIC_MEMORY, metadata={"event_type": "memory"})
            row = store._conn.execute(
                "SELECT COUNT(*) FROM memory_embeddings"
            ).fetchone()
            assert row[0] == 1
            store.delete_node(nid)
            row = store._conn.execute(
                "SELECT COUNT(*) FROM memory_embeddings"
            ).fetchone()
            assert row[0] == 0
        finally:
            store.close()


class TestVecStoreKeepsSideTable:
    def test_embeddings_dual_written_when_vec_available(self, tmp_path):
        store = _mk_store(tmp_path, "withvec.db")
        try:
            if not store._vec_available:
                pytest.skip("sqlite-vec unavailable in this environment")
            store.store(content=SEMANTIC_MEMORY, metadata={"event_type": "memory"})
            side = store._conn.execute(
                "SELECT COUNT(*) FROM memory_embeddings"
            ).fetchone()[0]
            vec = store._conn.execute("SELECT COUNT(*) FROM memories_vec").fetchone()[0]
            assert side == vec == 1
        finally:
            store.close()

    def test_store_survives_losing_the_extension(self, tmp_path, monkeypatch):
        """A db written with vec0 keeps serving vector search without it."""
        db = tmp_path / "migrating.db"
        store = SQLiteStore(db_path=str(db))
        if not store._vec_available:
            store.close()
            pytest.skip("sqlite-vec unavailable in this environment")
        nid = store.store(
            content=SEMANTIC_MEMORY, metadata={"event_type": "user_preference"}
        )
        store.store(content=DISTRACTORS[0].format(i=1), metadata={"event_type": "memory"})
        store.close()

        real_import = builtins.__import__

        def blocked(name, *args, **kwargs):
            if name == "sqlite_vec":
                raise ImportError("sqlite_vec blocked (env changed)")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", blocked)
        reopened = SQLiteStore(db_path=str(db))
        try:
            assert reopened._vec_available is False
            results = reopened.query(
                "what vegetables do I have for cooking at home?",
                limit=5,
                use_cache=False,
            )
            assert any(r.id == nid for r in results)
        finally:
            reopened.close()
