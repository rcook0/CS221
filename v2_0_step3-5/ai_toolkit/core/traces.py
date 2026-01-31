from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Generic, Hashable, List, Optional, Tuple, TypeVar

S = TypeVar("S", bound=Hashable)
TRACE_SCHEMA_VERSION = "2.0"

A = TypeVar("A")


@dataclass
class SearchTrace(Generic[S, A]):
    """Optional trace data captured during search.

    Notes
    -----
    - parent and g_score represent the best-known predecessor tree at termination.
    - expanded_order is the order states were expanded (popped from frontier).
    - generated_edges can be large; collect only when explicitly requested.
    """

    parent: Dict[S, Tuple[Optional[S], Optional[A]]]
    g_score: Dict[S, float]
    expanded_order: List[S]
    generated_edges: Optional[List[Tuple[S, S, A, float]]] = None


import json
import time
from pathlib import Path
from typing import Any

def write_search_trace_jsonl(
    out_path: str | Path,
    *,
    trace: "SearchTrace[Any, Any]",
    result: Any | None = None,
    context: dict[str, Any] | None = None,
    include_edges: bool = False,
    max_edges: int | None = None,
    toolkit_version: str | None = None,
) -> None:
    """Write a versioned JSONL trace for search runs.

    Records:
      - header: trace_schema + metadata
      - expand: expansion sequence (with best-known g)
      - edge (optional): generated edges (bounded by max_edges)
      - finish (optional): summary metrics and result schema version
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header = {
        "type": "header",
        "trace_schema": TRACE_SCHEMA_VERSION,
        "toolkit_version": toolkit_version,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "context": context or {},
    }
    if header["toolkit_version"] is None:
        header.pop("toolkit_version")

    with out_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(header, ensure_ascii=False) + "\n")

        for idx, s in enumerate(trace.expanded_order):
            g = trace.g_score.get(s, None)
            parent = trace.parent.get(s, (None, None))[0]
            rec = {
                "type": "expand",
                "idx": idx,
                "state": repr(s),
                "g": g,
                "parent": (repr(parent) if parent is not None else None),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        if include_edges and trace.generated_edges is not None:
            count = 0
            for (src, dst, a, step_cost) in trace.generated_edges:
                if max_edges is not None and count >= max_edges:
                    break
                rec = {
                    "type": "edge",
                    "src": repr(src),
                    "dst": repr(dst),
                    "action": repr(a),
                    "step_cost": float(step_cost),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                count += 1

        if result is not None:
            finish = {
                "type": "finish",
                "status": getattr(result, "status", None),
                "cost": getattr(result, "cost", None),
                "expanded": getattr(result, "expanded", None),
                "generated": getattr(result, "generated", None),
                "reopens": getattr(result, "reopens", None),
                "max_frontier": getattr(result, "max_frontier", None),
                "runtime_sec": getattr(result, "runtime_sec", None),
                "result_schema": getattr(result, "schema_version", None),
            }
            f.write(json.dumps(finish, ensure_ascii=False) + "\n")
