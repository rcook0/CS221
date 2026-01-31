from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Dict, Hashable, Iterable, Optional, Sequence, Set, Tuple, TypeVar

from .search import SearchResult, SearchTrace

S = TypeVar("S", bound=Hashable)
A = TypeVar("A")


def _dot_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', r'\"')


def write_search_dot(
    res: SearchResult[S, A],
    out_path: str | Path,
    *,
    title: str = "search",
    max_nodes: int = 2000,
    include_cost_in_labels: bool = True,
) -> Path:
    """Write a Graphviz DOT file for the search tree.

    - Nodes are the discovered states in the final parent tree.
    - Edges are the chosen parent pointers (i.e., a tree/forest).
    - The solution path (if any) is emphasized.

    Tip:
      dot -Tpng search.dot -o search.png
    """
    if res.trace is None:
        raise ValueError("SearchResult has no trace. Run the algorithm with trace=True.")

    trace = res.trace
    parent = trace.parent

    # Ensure we always include the actual solution path.
    path_states: Sequence[S] = res.states
    keep: Set[S] = set(path_states)

    # Add other nodes up to max_nodes.
    for s in parent.keys():
        if len(keep) >= max_nodes:
            break
        keep.add(s)

    expanded_set = set(trace.expanded_order)

    # Compute path edges for emphasis
    path_edges: Set[Tuple[S, S]] = set()
    for u, v in zip(path_states, path_states[1:]):
        path_edges.add((u, v))

    def node_id(s: S) -> str:
        # deterministic-ish, but safe: use repr as stable label and hash for id
        return f"n{abs(hash(s))}"

    lines = []
    lines.append("digraph Search {")
    lines.append("  rankdir=LR;")
    lines.append(f"  labelloc=\"t\";")
    lines.append(
        "  label=\""
        + _dot_escape(
            f"{title} | cost={res.cost} | expanded={res.expanded} | generated={res.generated} | reopens={res.reopens} | runtime={res.runtime_sec:.6f}s"
        )
        + "\";"
    )

    # Nodes
    for s in keep:
        g = trace.g_score.get(s)
        lbl = repr(s)
        if include_cost_in_labels and g is not None:
            lbl = f"{lbl}\\ng={g:.4g}"
        shape = "box" if s in expanded_set else "ellipse"
        periph = "2" if s == path_states[-1] else "1"
        lines.append(f"  {node_id(s)} [label=\"{_dot_escape(lbl)}\", shape={shape}, peripheries={periph}];")

    # Edges from the parent tree.
    for child, (par, act) in parent.items():
        if par is None or act is None:
            continue
        if par not in keep or child not in keep:
            continue
        # emphasize solution path
        attrs = []
        if (par, child) in path_edges:
            attrs.append("penwidth=3")
        label = _dot_escape(str(act))
        attrs.append(f"label=\"{label}\"")
        attr_str = ", ".join(attrs)
        lines.append(f"  {node_id(par)} -> {node_id(child)} [{attr_str}];")

    lines.append("}")

    out_path = Path(out_path)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def write_search_trace_jsonl(
    trace: SearchTrace[S, A],
    out_path: str | Path,
    *,
    max_edges: int = 1_000_000,
) -> Path:
    """Write a simple JSONL trace.

    Events:
      - {"type":"expand", "state":..., "idx": i}
      - {"type":"edge", "src":..., "dst":..., "action":..., "cost":...}

    Use with trace_edges=True to get edge events.
    """
    out_path = Path(out_path)
    with out_path.open("w", encoding="utf-8") as f:
        for i, s in enumerate(trace.expanded_order):
            f.write(json.dumps({"type": "expand", "idx": i, "state": repr(s)}) + "\n")

        if trace.generated_edges is not None:
            n = 0
            for src, dst, act, cost in trace.generated_edges:
                f.write(
                    json.dumps(
                        {
                            "type": "edge",
                            "src": repr(src),
                            "dst": repr(dst),
                            "action": str(act),
                            "cost": float(cost),
                        }
                    )
                    + "\n"
                )
                n += 1
                if n >= max_edges:
                    break
    return out_path
