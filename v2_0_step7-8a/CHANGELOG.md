# Changelog

## 2.0.0
- Internal-library milestone: stable `ai_toolkit.core` APIs (protocols, results, traces, RNG).
- Filesystem refactor into `core/`, `domains/`, `viz/`, `cli/`, `experimental/`.
- Versioned schemas for results (`2.0`), traces (`2.0`), and bench rows (`2.0`).
- CLI updated to call `ai_toolkit.core.*` entrypoints and emit schema-tagged artifacts.
- Test suite expanded: search/MDP property tests, schema validation tests, and CLI smoke tests.
- Backwards-compatible re-export modules kept for v1.x import paths (deprecated).

