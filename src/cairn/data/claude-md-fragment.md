<!-- Cairn:BEGIN — managed by cairn setup, do not edit this block -->
## Memory (Cairn)

You have Cairn persistent memory. At session start:
1. Call `cairn_welcome()` for context briefing
2. Call `cairn_protocol()` for your operating instructions — it's your coordination playbook
3. Follow the protocol it returns

Quick reference (protocol has full details):
- `[MEMORY]`/`[HANDOFF]`/`[COORD]` blocks from hooks = ground truth
- Before non-trivial tasks: `cairn_query()` for prior context
- Before spawning subagents: `cairn_query()` first, inject results into agent prompt (subagents can't call Cairn)
- After completing tasks: `cairn_store(content, "decision")` for key outcomes — minimum 1 store per session
- User says "remember": `cairn_store(text, "user_preference")`
- Context getting full: `cairn_checkpoint` to save state
- Load user context: `cairn_profile()` after welcome/protocol
- Before architecture decisions: `cairn_reflect(action="evolution", topic=<domain>)` to check prior thinking
- After `cairn_store`: check `cairn_memory(similar)` and link related memories to build the knowledge graph
- NEVER fabricate URLs — read from files, query Cairn, or verify via web fetch

If Cairn is unavailable, use basic coordination:
- Before state changes: check `git log` and ask before deploying
- After tasks: store decisions with `cairn_store()`
<!-- Cairn:END -->
