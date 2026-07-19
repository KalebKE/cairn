"""End-to-end MCP protocol test.

Spawns the real `python -m cairn.server.mcp_server` over stdio and drives the
JSON-RPC handshake (initialize -> tools/list -> tools/call) through the mcp
client SDK. This exercises the ~900-LOC dispatch layer in mcp_server.py that
handler-level unit tests never touch — the two prior "mirror tests" in
test_http_transport.py reimplemented production logic and would stay green even
if the server broke.

Uses CAIRN_SKIP_EMBEDDINGS=1 (hash-vector fallback) so the subprocess starts
fast and needs no ONNX model cache, and CAIRN_CONDENSED=0 so cairn_store /
cairn_query are exposed directly rather than behind the cairn_call meta-tool.
"""
import os
import shutil
import sys
import tempfile

import pytest


async def _drive(tmp_home):
    from mcp.client.stdio import stdio_client, StdioServerParameters
    from mcp.client.session import ClientSession

    env = {
        **os.environ,
        "CAIRN_HOME": str(tmp_home),
        "CAIRN_SKIP_EMBEDDINGS": "1",
        "CAIRN_CONDENSED": "0",
        "CAIRN_IDLE_TIMEOUT": "0",
        "CAIRN_MAINT_INTERVAL_S": "0",
        "CAIRN_TRANSPORT": "stdio",
    }
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "cairn.server.mcp_server"],
        env=env,
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            names = {t.name for t in tools.tools}
            assert "cairn_store" in names, f"cairn_store missing from {names}"
            assert "cairn_query" in names

            marker = "e2e-protocol-marker-xyzzy quantum ferret cascade"
            store_res = await session.call_tool(
                "cairn_store",
                {"content": f"Decision: {marker} is the canary for the MCP e2e test",
                 "event_type": "decision"},
            )
            assert not store_res.isError, store_res.content
            store_text = "".join(getattr(c, "text", "") for c in store_res.content)
            assert "mem-" in store_text or "Stored" in store_text

            query_res = await session.call_tool(
                "cairn_query", {"query": marker, "limit": 5},
            )
            assert not query_res.isError, query_res.content
            query_text = "".join(getattr(c, "text", "") for c in query_res.content)
            assert marker.split()[0] in query_text or "ferret" in query_text


@pytest.mark.asyncio
async def test_mcp_stdio_initialize_list_and_roundtrip():
    """Full JSON-RPC roundtrip against the real server subprocess.

    Home lives under /tmp (short path) so the daemon's AF_UNIX hook socket
    (CAIRN_HOME/hook.sock) stays under the ~104-char limit.
    """
    home = tempfile.mkdtemp(prefix="cx", dir="/tmp")
    try:
        await _drive(home)
    except FileNotFoundError:
        pytest.skip("mcp stdio client transport unavailable in this environment")
    finally:
        shutil.rmtree(home, ignore_errors=True)
