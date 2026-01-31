# ai_toolkit Roadmap

## Status
**v2.0 COMPLETE**

All v2.0 objectives are satisfied:
- Stable internal library (`core/*`)
- Versioned schemas (results, traces, bench)
- Deterministic execution (seeded RNG)
- CLI wired to core entrypoints
- Property tests, schema tests, CLI smoke tests
- Linting, typing, CI, docs, migration notes
- Public API explicitly declared

v2.0 is now a safe dependency.

---

## v2.0 (Internal Library Milestone) — DONE

### Scope delivered
- **Architecture**
  - Clear separation: `core / domains / viz / cli / experimental`
  - Backwards-compatible re-exports for v1.5 users
- **Core protocols**
  - `SearchProblem`, `MDP`, `ZeroSumGame`
- **Algorithms**
  - Search: BFS, UCS, A* (with reopen semantics)
  - MDPs: Value Iteration, Policy Iteration
  - Games: Minimax, Alpha–Beta
  - Optimization / DP / ML primitives
- **Determinism**
  - Central RNG control
  - Deterministic tie-breaking
- **Schemas**
  - Result schema v2.0
  - Trace schema v2.0 (JSONL + HTML)
  - Bench schema v2.0
- **Hardening**
  - Property tests (A* vs UCS, VI vs PI)
  - Schema validation tests
  - CLI smoke tests
  - Ruff + mypy (strict on core)
  - CI pipeline
- **Documentation**
  - README with Public API section
  - Migration guide (v1.5 → v2.0)
  - API stability guarantees

**Outcome:** ai_toolkit is a clean, stable internal library.

---

## v2.1–v2.4 (Executable Textbook Lane)

### v2.1 — Domain Suite v1
- GridWorld (heuristics + rendering)
- 8-puzzle (Manhattan, misplaced tiles)
- Word ladder / edit graph
- Matrix games (for minimax / mixed strategies)
- Each domain includes:
  - Reference heuristics
  - Baseline optimal solvers
  - Deterministic generators

### v2.2 — Chapter Runners
- `python -m ai_toolkit.textbook.run <chapter>`
- Emits:
  - Bench tables (CSV/JSONL)
  - HTML trace replays
  - Auto-filled Markdown summaries

### v2.3 — HTML Dashboards
- Offline, self-contained dashboards per chapter
- Algorithm comparison + trace replay
- No external JS dependencies

### v2.4 — Assessment Mode
- Auto-generated checkpoint questions
- Short answer keys derived from tool output
- Aligns with CS221-style course reader

---

## v2.5+ (Research Sandbox Lane)

### Scope
- Experimental algorithms only:
  - Weighted A*, IDA*, MCTS
  - Loopy BP, advanced MCMC
  - RL with function approximation

### Graduation rules
An algorithm may move from `experimental/` to `core/` only if:
1. Stable interface defined
2. Property + regression tests added
3. Bench evidence collected
4. Failure modes documented

---

## Guiding Principles
- **Core > correctness > determinism**
- **Textbook > clarity > reproducibility**
- **Sandbox > velocity > isolation**

This roadmap is intentionally conservative: new power enters through evidence, not enthusiasm.