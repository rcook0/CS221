from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

from ..domains.tram import TramCosts, TransportationProblem, shortest_cost_dp
from ..search import SearchResult, astar, bfs, dfs, ucs


@dataclass(frozen=True)
class BenchmarkRow:
    """A single benchmark observation.

    `params` is intentionally a dict to make it easy to extend domains without
    changing this container.
    """

    domain: str
    algo: str
    params: Dict[str, Any]
    metrics: Dict[str, Any]


def _result_to_metrics(res: SearchResult[Any, Any]) -> Dict[str, Any]:
    return {
        "cost": res.cost,
        "plan_len": len(res.actions),
        "expanded": res.expanded,
        "generated": res.generated,
        "reopens": res.reopens,
        "max_frontier": res.max_frontier,
        "runtime_sec": res.runtime_sec,
    }


def benchmark_tram_search(
    Ns: Sequence[int],
    *,
    walk_cost: float = 1.0,
    tram_cost: float = 2.0,
    algos: Sequence[str] = ("ucs", "astar"),
    repeats: int = 1,
    max_expansions: int = 10_000_000,
) -> List[BenchmarkRow]:
    """Benchmark search algorithms on the walk-vs-tram domain.

    The harness records algorithm stats (expanded/generated/reopens/frontier/runtime)
    and also includes the DP-optimal cost as a correctness reference.
    """

    rows: List[BenchmarkRow] = []

    for N in Ns:
        costs = TramCosts(walk=float(walk_cost), tram=float(tram_cost))
        problem = TransportationProblem(int(N), costs=costs)
        dp_cost, _hist = shortest_cost_dp(problem)

        for _ in range(int(repeats)):
            for algo in algos:
                algo = algo.lower().strip()
                if algo == "bfs":
                    res = bfs(problem, max_expansions=max_expansions)
                elif algo == "dfs":
                    res = dfs(problem, max_expansions=max_expansions)
                elif algo == "ucs":
                    res = ucs(problem, max_expansions=max_expansions)
                elif algo in ("astar", "a*", "a_star"):
                    res = astar(problem, heuristic=problem.admissible_heuristic, max_expansions=max_expansions)
                    algo = "astar"
                else:
                    raise ValueError(f"Unknown algo: {algo}")

                params = {
                    "N": int(N),
                    "walk_cost": float(walk_cost),
                    "tram_cost": float(tram_cost),
                }
                metrics = _result_to_metrics(res)
                metrics["dp_opt_cost"] = float(dp_cost)
                metrics["optimality_gap"] = float(res.cost) - float(dp_cost)

                rows.append(BenchmarkRow(domain="tram", algo=algo, params=params, metrics=metrics))

    return rows


def to_jsonl(rows: Iterable[BenchmarkRow], fp) -> None:
    """Write benchmarks as JSON Lines (one object per line)."""

    for r in rows:
        obj = asdict(r)
        fp.write(json.dumps(obj, sort_keys=True))
        fp.write("\n")


def to_csv(rows: Sequence[BenchmarkRow], fp) -> None:
    """Write a flat CSV. (Good for quick spreadsheeting.)"""

    # Flatten with stable column set.
    # We keep params.* and metrics.* as their own columns.
    import csv

    all_param_keys = sorted({k for r in rows for k in r.params.keys()})
    all_metric_keys = sorted({k for r in rows for k in r.metrics.keys()})

    fieldnames = ["domain", "algo"] + [f"param.{k}" for k in all_param_keys] + [f"metric.{k}" for k in all_metric_keys]

    w = csv.DictWriter(fp, fieldnames=fieldnames)
    w.writeheader()
    for r in rows:
        out: Dict[str, Any] = {"domain": r.domain, "algo": r.algo}
        for k in all_param_keys:
            out[f"param.{k}"] = r.params.get(k)
        for k in all_metric_keys:
            out[f"metric.{k}"] = r.metrics.get(k)
        w.writerow(out)
