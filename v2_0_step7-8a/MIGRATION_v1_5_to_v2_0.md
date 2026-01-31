# Migration guide: v1.5 -> v2.0

v2.0 introduces a stable internal library surface under `ai_toolkit.core` and a clearer package split:
- `ai_toolkit.core`: algorithms + protocols + results/traces (stable)
- `ai_toolkit.domains`: reference domains (tram, etc.)
- `ai_toolkit.viz`: HTML trace rendering and other visualization helpers
- `ai_toolkit.cli`: CLI implementation
- `ai_toolkit.experimental`: sandbox (not stable)

## Import changes

### Search
Old:
```python
from ai_toolkit.search import astar, ucs
```

New (preferred):
```python
from ai_toolkit.core.search.algorithms import astar, ucs
```

### MDP
Old:
```python
from ai_toolkit.mdp import value_iteration
```

New:
```python
from ai_toolkit.core.mdp.algorithms import value_iteration
```

### Results and schemas
Old (implicit dict/tuple-ish):
```python
res = astar(...)
print(res.cost)
```

New:
```python
from ai_toolkit.core.results import SearchResult
res = astar(...)
print(res.cost, res.schema_version, res.meta)
```

## CLI
`python -m ai_toolkit ...` remains supported. The CLI now calls core entrypoints and emits schema-tagged JSONL/CSV.

## Deprecation window
The `ai_toolkit.search`, `ai_toolkit.mdp`, etc. modules are retained as re-exports for compatibility, but should be considered deprecated.
