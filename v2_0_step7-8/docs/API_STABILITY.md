# API stability (v2.0)

## Stable surface
The following modules are considered stable in v2.0.x:

- `ai_toolkit.core.protocols`
- `ai_toolkit.core.results`
- `ai_toolkit.core.traces`
- `ai_toolkit.core.rng`
- `ai_toolkit.core.search.algorithms`
- `ai_toolkit.core.mdp.algorithms`
- `ai_toolkit.core.games.algorithms`

### Stability guarantees
- No breaking changes to the public names, signatures, or schema fields in v2.0.x.
- Schema versions (`results`, `traces`, `bench`) remain `2.0` for the entire v2.0 line.
- Deprecations may be introduced, but removals occur only in the next major version.

## Not stable
- `ai_toolkit.experimental/*`
- any module/attribute starting with `_`
- `legacy/*`
