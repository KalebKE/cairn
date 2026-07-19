# Troubleshooting

Common issues and solutions when using Cairn.

---

## Model Download Fails

**Symptom**: First query hangs or errors with "Failed to download model" or connection timeout.

**Cause**: Cairn downloads the bge-small-en-v1.5 ONNX model (~90 MB) on first use to `~/.cache/cairn/models/`.

**Solutions**:

1. **Proxy or firewall**: If behind a corporate proxy, set `HTTPS_PROXY` before starting the MCP server:
   ```bash
   export HTTPS_PROXY=http://proxy.example.com:8080
   ```

2. **Disk space**: Ensure at least 200 MB free in `~/.cache/cairn/`.

3. **Manual download**: Download the model files manually from [Hugging Face](https://huggingface.co/BAAI/bge-small-en-v1.5) and place them in `~/.cache/cairn/models/bge-small-en-v1.5-onnx/`.

4. **Verify with doctor**: Run `cairn doctor` to check model status.

---

## ONNX Runtime Not Found

**Symptom**: `ImportError: onnxruntime not found` or embedding generation fails.

**Cause**: The `onnxruntime` package is missing from your Python environment.

**Solution**:

```bash
pip install onnxruntime
```

Or install Cairn with all dependencies:

```bash
pip install "cairn[all]"
```

Note: CoreML acceleration is intentionally disabled due to a memory leak in Apple's ANE runtime. CPU-only inference is used.

---

## sqlite-vec Not Available

**Symptom**: Warning about "sqlite-vec not available, using hash-based fallback" or degraded search quality.

**Cause**: The `sqlite-vec` extension (used for vector similarity search) isn't installed or can't be loaded.

**Impact**: Cairn falls back to hash-based approximate nearest neighbors. Search still works but with lower accuracy for semantic queries.

**Solution**:

```bash
pip install sqlite-vec
```

If `pip install` fails (e.g., no wheel for your platform), Cairn will continue to function with the fallback. Run `cairn doctor` to verify the status.

---

## High RSS Memory (300-400 MB)

**Symptom**: `cairn_health` reports "critical" RSS memory at 300-400 MB, or system monitor shows high memory usage.

**Cause**: This is **expected behavior**, not a memory leak. The ONNX embedding model loads ~300 MB into RAM on first semantic query.

**Lifecycle**:

```
~31 MB idle → ~337 MB after first query → ~31 MB after 10 min idle
```

The model auto-unloads after 10 minutes without queries. The health check critical threshold is set to 800 MB to avoid false alarms during normal peak usage.

**When to worry**: If RSS stays above 500 MB with no active queries for more than 15 minutes, or if it grows unboundedly over time, that may indicate an actual issue. File a bug report with the output of `cairn_health`.

---

## Database Locked

**Symptom**: `sqlite3.OperationalError: database is locked`

**Cause**: Multiple processes are trying to write to `~/.cairn/cairn.db` simultaneously. SQLite handles concurrent reads but serializes writes.

**Solutions**:

1. **Check for stuck processes**:
   ```bash
   ps aux | grep cairn
   ```
   Kill any orphaned Cairn server processes.

2. **WAL mode**: Cairn uses WAL mode by default, which allows concurrent reads during writes. If your database isn't in WAL mode:
   ```bash
   cairn doctor
   ```
   This will report the journal mode and fix it if needed.

---

## Hook Server Not Running

**Symptom**: Claude Code hooks don't trigger Cairn (no memory capture, no coordination).

**Cause**: The hook daemon (`hook_server.py`) isn't running or can't be reached via its Unix Domain Socket.

**Solutions**:

1. **Check status**:
   ```bash
   cairn doctor
   ```
   Look for the "Hook server" line.

2. **Restart manually**:
   ```bash
   cairn hooks restart
   ```

3. **Check socket**: The UDS lives at `~/.cairn/hook.sock`. If it's stale (process died without cleanup), remove it:
   ```bash
   rm ~/.cairn/hook.sock
   cairn hooks start
   ```

Note: Hooks fail open — if the daemon is unreachable, Claude Code continues working normally. You just lose auto-capture and coordination features until the daemon is restarted.

---

## Setup Issues

For initial setup problems, run the diagnostic tool:

```bash
cairn doctor
```

This checks:
- Python version compatibility
- Required and optional dependencies
- Database status and schema version
- Model availability
- Hook server status
- Disk space and permissions
