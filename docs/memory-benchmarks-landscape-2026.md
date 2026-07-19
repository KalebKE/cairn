# Agent Memory Benchmarks Landscape & MemoryArena Analysis

> Research conducted Feb 20, 2026. MemoryArena: arxiv 2602.16313 (Feb 18, 2026).

## 1. The Core Claim: Recall != Agentic Performance

MemoryArena's central finding: systems that score near-saturated on recall benchmarks (LoCoMo, LongMemEval) achieve **0-23% success rate** on interdependent multi-session agentic tasks. The gap is confirmed by multiple independent efforts:

| Paper | Date | Key Finding |
|-------|------|-------------|
| **MemoryArena** (He et al.) | Feb 2026 | 4 domains, 766 subtasks, best SR 23%. Mem0/Letta/ReasoningBank all 0% on web shopping. |
| **Mem2ActBench** (Shen et al.) | Jan 2026 | Tests proactive memory-to-action in tool-based tasks. Same thesis: recall != action. |
| **Evo-Memory** (Wei et al., DeepMind) | Nov 2025 | Streaming benchmark for self-evolving memory. Tests learning from mistakes across sequential tasks. |
| **ReasoningBank** (Ouyang et al., Google Cloud) | Sep 2025 | Distills generalizable reasoning strategies. Tested in MemoryArena: 0% SR on shopping/travel. |

**Key insight**: The field is converging on "memory recall benchmarks are insufficient" as consensus in 2025-2026.

## 2. Landscape of Memory Benchmarks (Chronological)

### Recall-Only Benchmarks (2024-2025)
- **LoCoMo** (Maharana, 2024): Long-context conversational QA. 7,512 questions. Near-saturated by modern systems.
- **LongMemEval** (Wu, 2025): 500 questions, 5 abilities. Cairn: 95.4%. Tests recall, temporal reasoning, knowledge updates. No agentic actions.
- **MemoryAgentBench** (Hu, 2025): Incremental multi-turn info intake. 2k queries. Still recall-focused.
- **MemoryBench** (Ai, 2025): 778 queries testing memory and continual learning. QA format.

### Action-Only Benchmarks (no persistent memory needed)
- **WebArena** (Zhou): Single-session web tasks. 812 tasks.
- **SWE-Bench** (Jimenez, 2023): Single-session code fixes.

### Bridging Recall + Action (2025-2026): The New Wave
- **Evo-Memory** (DeepMind, Nov 2025): Sequential tasks from existing benchmarks fed in streaming. No explicit cross-task dependencies.
- **Mem2ActBench** (Jan 2026): Proactive memory-to-action in tool-based tasks.
- **AgencyBench** (Li et al., 2026): 1M-token real-world contexts, 138 tasks with some interdependency.
- **MemoryArena** (Feb 2026): Most rigorous. Human-crafted causal dependencies between subtasks. 766 subtasks. POMDP framing.

## 3. Related Memory Systems & Architectures

### Hindsight (Latimer et al., Dec 2025) -- MOST RELEVANT TO Cairn
- Paper: "Hindsight is 20/20: Building Agent Memory that Retains, Recalls, and Reflects"
- Code: github.com/vectorize-io/hindsight (open source, MCP server)
- **TEMPR** (Temporal Entity Memory Priming Retrieval): Stores narrative facts in a memory graph with 4 link types: temporal, semantic, entity, causal. Retrieval fuses semantic + keyword + graph + temporal search.
- **CARA** (Coherent Adaptive Reasoning Agents): Maintains a behavioral profile with disposition parameters (skepticism, literalism, empathy). Updates beliefs coherently.
- Claims **83.6% on LongMemEval** with a 20B open-source model (vs 39% baseline with same model).
- **Relevance**: Their memory graph with causal links is what MemoryArena says is missing. Cairn stores flat facts; Hindsight stores facts with relationship structure. Their CARA reflection mechanism is similar to Cairn's protocol but more formalized.

### ReasoningBank (Google Cloud, Sep 2025)
- Distills generalizable reasoning strategies from successful agent runs.
- Stores reasoning patterns, not just facts.
- Performs poorly in MemoryArena (0% SR on structured tasks), suggesting reasoning memory alone is insufficient.

### Oracle's CoALA-Based Taxonomy (Feb 2026 blog)
- 4 memory types: working, procedural, semantic, episodic.
- Key quote: "VentureBeat predicts contextual memory will surpass RAG for agentic AI in 2026."
- Sleep-time computation (background consolidation) highlighted as next frontier.

### AWS Agent Evaluation Framework (Feb 2026)
- Memory metric is narrow: "context retrieval accuracy" only.
- Does not test cross-session memory application.
- Validates MemoryArena's point: even enterprise eval frameworks haven't caught up.

## 4. Community Sentiment (X/Twitter, Reddit)

### X/Twitter
- **@omarsar0 (DAIR.AI)**: Amplified MemoryArena with "Agent memory benchmarks are misleading." High engagement.
- No significant pushback found yet (paper is 2 days old).

### Reddit Themes
- **r/AI_Agents**: Users identify 3 levels: L1 (within-session), L2 (cross-session facts), L3 (cross-session identity/preferences). "Most memory solutions fail because they treat it as a retrieval problem."
- **r/AI_Agents**: "Until agents can gracefully update what they think is true, long-term memory will always create inertia." Echoes belief-drift finding.
- **r/LocalLLaMA**: Someone benchmarked providers using LOCOMO. Exactly the kind of evaluation MemoryArena argues is insufficient.
- **r/AI_Agents**: "2 years building agent memory systems, ended up just using Git." Pragmatic: structured state > sophisticated retrieval.

### General Sentiment
Main frustrations:
1. Repeated onboarding / forgetting across sessions
2. Inability to maintain consistent preferences over time
3. Memory inertia (outdated beliefs not updating)
4. Retrieval returning semantically similar but causally irrelevant content

## 5. What Cairn Can Learn

### Confirmed Strengths
1. **Category-typed storage** (decision, preference, user_preference) is a step toward structured memory.
2. **Protocol system** enforces structured memory interaction (query before acting, store after deciding). Most systems lack this.
3. **Checkpoint/resume** preserves execution context, not just facts.

### Identified Gaps
1. **No causal/dependency links between memories**: When decision B depends on decision A, Cairn doesn't model that relationship. Hindsight's TEMPR does this with 4 link types.
2. **Retrieval is similarity-based, not state-aware**: MemoryArena's POMDP framing says memory should provide sufficient statistics for belief-state estimation. Cairn's semantic search can miss causally relevant but semantically distant facts.
3. **No belief/state tracking**: Cairn stores what happened but doesn't maintain an evolving model of "what is currently true." Hindsight's CARA does this.
4. **No reflection/consolidation loop**: No background process that reviews memories, resolves conflicts, or compresses redundant entries. Oracle's blog highlights "sleep-time computation" as the next frontier. (Note: Cairn has consolidation phases 0-2, but these are decay/prune, not semantic reflection.)

### Feature Ideas (Ranked by Impact vs Effort)
1. **Dependency-aware storage** (HIGH impact, MEDIUM effort): When storing a decision, optionally link it to prior memories it depends on. `cairn_store(content, "decision", depends_on=["mem_id_123"])`
2. **State tracking type** (HIGH impact, HIGH effort): A new memory type "state" that gets UPDATED (not appended) as tasks progress. Represents current belief about an ongoing situation.
3. **Causal retrieval mode** (MEDIUM impact, MEDIUM effort): When querying, optionally traverse dependency links to surface the full chain, not just top-K similar.
4. **Background consolidation** (MEDIUM impact, HIGH effort): Periodic process that reviews recent memories, merges duplicates, resolves conflicts. "Sleep-time compute."

## 6. Planned Experiments

### A. Dependency Chain Test (LOW cost, HIGH signal)
Create 10 chains of 4-6 interdependent decisions. Test if cairn_query surfaces the right chain when given only the final subtask.

### B. Belief Drift Test (LOW cost, MEDIUM signal)
Store a fact, then store contradictions. Query: does Cairn return latest or both?

### C. Cross-Session Action Test (MEDIUM cost, HIGH signal) -- DEFERRED
Simulate MemoryArena-style multi-session shopping tasks. Requires more setup.

### D. Semantic vs Causal Retrieval (LOW cost, HIGH signal)
Store 20 memories with deliberate semantic/causal mismatches. Measure false positive and false negative rates.

### E. Run Cairn on MemoryArena (when code drops) -- DEFERRED
Watch memoryarena.github.io for public release.

## 7. Notes

- **MemoryArena code/data**: NOT public yet. Website says "coming soon." Watch memoryarena.github.io.
- **Awesome-AI-Memory repo**: Awaiting response from maintainers before interacting.
- **Hindsight code**: Public at github.com/vectorize-io/hindsight. MCP server. Worth exploring as reference implementation.

---

## Experiment Status

Experiments A (Dependency Chain), B (Belief Drift), and D (Semantic vs Causal) **deferred**. Rationale: the research already confirms the gaps theoretically, and injecting test data into production memory adds noise without proportional insight.

**Waiting on:** MemoryArena public code/data release (memoryarena.github.io). When available, run Cairn as memory backend against their GPT-5.1-mini task agent on Progressive Web Search and Formal Reasoning domains.

**Next actionable step:** Scope `depends_on` parameter for cairn_store (feature idea #1: dependency-aware storage). Highest impact-to-effort ratio for closing the causal linking gap.
