<!-- Cairn:BEGIN — managed by cairn setup, do not edit this block -->
## Memory (Cairn)

You have Cairn persistent memory. At session start:
1. Call `cairn_welcome()` for context briefing
2. Call `cairn_protocol()` for your operating instructions — it's your coordination playbook
3. Follow the protocol it returns

Quick reference (protocol has full details):
- `[MEMORY]`/`[HANDOFF]`/`[COORD]` blocks from hooks = ground truth
- Before non-trivial tasks: `cairn_query()` for prior context
- After completing tasks: `cairn_store(content, "decision")` for key outcomes
- User says "remember": `cairn_store(text, "user_preference")`
- Context getting full: `cairn_checkpoint` to save state
- Load user context: `cairn_profile()` after welcome/protocol
- Before architecture decisions: `cairn_reflect(action="evolution", topic=<domain>)` to check prior thinking
- After `cairn_store`: check `cairn_memory(similar)` and link related memories to build the knowledge graph

### Multi-Agent Coordination
- Check `cairn_inbox()` for unread peer messages early in sessions
- Announce intent: `cairn_intent_announce(description="<goal>")` before starting work
- Before editing shared files: `cairn_file_check(file_path=...)` for conflicts
- After significant work: `cairn_task_complete(task_id=..., result="summary")`
- Before deploy/force-push: `cairn_action_check()` then `cairn_action_claim()` (atomic gate)
- Never `git add .` — always `git add <specific files>`

If Cairn is unavailable, use basic coordination:
- Before state changes: check `git log` and ask before deploying
- Never send emails, post tweets, or take externally-visible actions without explicit approval
- Commit only files you modified; `git add <files>` never `git add .`
- After tasks: store decisions with `cairn_store()`
<!-- Cairn:END -->
