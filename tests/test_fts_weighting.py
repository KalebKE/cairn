"""Two-column FTS (schema v16): trigger sync + weighted anticipated terms."""
import os
from unittest.mock import patch

import pytest


def _fts_match(store, term):
    return {
        r[0] for r in store._conn.execute(
            "SELECT m.node_id FROM memories_fts f JOIN memories m ON f.rowid = m.id "
            "WHERE memories_fts MATCH ?", (term,)
        ).fetchall()
    }


@pytest.fixture
def fts_store(store):
    if not getattr(store, "_fts_available", False):
        pytest.skip("FTS5 unavailable")
    return store


class TestTwoColumnFTS:
    def test_fresh_store_indexes_both_columns(self, fts_store):
        nid = fts_store.store(content="Deployment pipeline uses blue-green rollout strategy")
        fts_store._conn.execute(
            "UPDATE memories SET extracted_keywords = 'kubernetes failover' WHERE node_id = ?",
            (nid,),
        )
        fts_store._commit()
        assert nid in _fts_match(fts_store, '"rollout"')       # content column
        assert nid in _fts_match(fts_store, '"kubernetes"')    # keywords column

    def test_keyword_update_trigger_resyncs(self, fts_store):
        """The v16 AU trigger fires on extracted_keywords changes — the sync
        enrichment relies on (no manual external-content resync)."""
        nid = fts_store.store(content="User grows tomatoes and basil in the backyard")
        assert nid not in _fts_match(fts_store, '"homegrown"')
        fts_store._conn.execute(
            "UPDATE memories SET extracted_keywords = 'homegrown ingredients dinner' "
            "WHERE node_id = ?", (nid,),
        )
        fts_store._commit()
        assert nid in _fts_match(fts_store, '"homegrown"')
        assert nid in _fts_match(fts_store, '"tomatoes"')  # content survives the resync

    def test_delete_trigger_cleans_both_columns(self, fts_store):
        nid = fts_store.store(content="Ephemeral note about websocket reconnect backoff")
        fts_store._conn.execute(
            "UPDATE memories SET extracted_keywords = 'transient retry jitter' WHERE node_id = ?",
            (nid,),
        )
        fts_store._commit()
        fts_store.delete_node(nid)
        assert nid not in _fts_match(fts_store, '"websocket"')
        assert nid not in _fts_match(fts_store, '"jitter"')

    def test_kw_weight_downranks_keyword_only_matches(self, fts_store):
        """With CAIRN_FTS_KW_WEIGHT < 1, a content match must outrank a
        keyword-only match for the same term — the rank-1 protection the
        column split exists for."""
        content_hit = fts_store.store(
            content="Postgres connection pooling configuration for the billing service"
        )
        kw_hit = fts_store.store(
            content="Weekly team retro notes: velocity discussion and action items"
        )
        fts_store._conn.execute(
            "UPDATE memories SET extracted_keywords = "
            "'postgres connection pooling setup how to configure pooling' WHERE node_id = ?",
            (kw_hit,),
        )
        fts_store._commit()

        def top_node(kw_weight):
            with patch.dict(os.environ, {"CAIRN_FTS_KW_WEIGHT": str(kw_weight)}):
                results = fts_store._text_search("postgres connection pooling", limit=5)
            return results[0].id if results else None

        assert top_node(0.1) == content_hit
        # Both must still be *findable* at low weight (recall preserved)
        with patch.dict(os.environ, {"CAIRN_FTS_KW_WEIGHT": "0.1"}):
            ids = {r.id for r in fts_store._text_search("postgres connection pooling", limit=5)}
        assert {content_hit, kw_hit} <= ids

    def test_v15_single_column_store_migrates(self, tmp_cairn_dir):
        """A v15-shaped DB (single-column FTS, old triggers) upgrades to the
        two-column layout with the index repopulated."""
        from cairn.sqlite_store import SQLiteStore

        db = tmp_cairn_dir / "legacy.db"
        s = SQLiteStore(db_path=db)
        if not getattr(s, "_fts_available", False):
            s.close()
            pytest.skip("FTS5 unavailable")
        nid = s.store(content="Legacy row about gradle build cache misconfiguration")
        # Downgrade to the v15 shape
        c = s._conn
        c.execute("DROP TABLE memories_fts")
        c.execute("DROP TRIGGER IF EXISTS memories_ai")
        c.execute("DROP TRIGGER IF EXISTS memories_ad")
        c.execute("DROP TRIGGER IF EXISTS memories_au")
        c.execute(
            "CREATE VIRTUAL TABLE memories_fts "
            "USING fts5(content, content='memories', content_rowid='id')"
        )
        c.execute(
            "INSERT INTO memories_fts(rowid, content) "
            "SELECT id, content || ' ' || COALESCE(extracted_keywords, '') FROM memories"
        )
        c.execute("UPDATE schema_version SET version = 15")
        s._commit()
        s.close()

        s2 = SQLiteStore(db_path=db)
        try:
            cols = [r[1] for r in s2._conn.execute("PRAGMA table_info(memories_fts)").fetchall()]
            assert "extracted_keywords" in cols
            assert nid in _fts_match(s2, '"gradle"')  # repopulated
            version = s2._conn.execute("SELECT version FROM schema_version").fetchone()[0]
            assert version >= 16
        finally:
            s2.close()
