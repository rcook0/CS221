from __future__ import annotations

import argparse
import sys
from typing import List, Sequence

from .bench import benchmark_tram_search, to_csv, to_jsonl
from ..domains.tram import TramCosts, TransportationMDP, TransportationProblem, render_tram_grid
from ..core.mdp.algorithms import policy_iteration, value_iterationationation
from ..core.search.algorithms import SearchResult, astar, bfs, dfs, ucs
from ..viz import write_search_dot, write_search_trace_html, write_search_trace_jsonl


def _parse_int_list(spec: str) -> List[int]:
    """Parse either:

    - comma list: "10,20,30"
    - range spec: "start:end:step" (end is inclusive)

    Examples:
      "5:50:5" -> [5,10,15,...,50]
      "8" -> [8]
    """
    s = spec.strip()
    if not s:
        raise ValueError("Empty list spec")

    if ":" in s:
        parts = s.split(":")
        if len(parts) not in (2, 3):
            raise ValueError(f"Bad range spec: {spec!r}")
        start = int(parts[0])
        end = int(parts[1])
        step = int(parts[2]) if len(parts) == 3 else 1
        if step == 0:
            raise ValueError("step must be non-zero")
        if start <= end and step < 0:
            raise ValueError("step must be positive for ascending ranges")
        if start >= end and step > 0:
            # allow descending only if step is negative
            raise ValueError("step must be negative for descending ranges")

        out: List[int] = []
        if step > 0:
            x = start
            while x <= end:
                out.append(x)
                x += step
        else:
            x = start
            while x >= end:
                out.append(x)
                x += step
        return out

    if "," in s:
        return [int(p.strip()) for p in s.split(",") if p.strip()]

    return [int(s)]


def _print_search_result(res: SearchResult[object, object]) -> None:
    print(f"cost: {res.cost}")
    print(f"plan_len: {len(res.actions)}")
    print(f"expanded: {res.expanded}")
    print(f"generated: {res.generated}")
    print(f"reopens: {res.reopens}")
    print(f"max_frontier: {res.max_frontier}")
    print(f"runtime_sec: {res.runtime_sec:.6f}")


def _cmd_run_tram(args: argparse.Namespace) -> int:
    costs = TramCosts(walk=float(args.walk_cost), tram=float(args.tram_cost))
    problem = TransportationProblem(int(args.N), costs=costs)

    want_trace = bool(args.dot_out) or bool(args.trace_out) or bool(args.html_out)
    trace_edges = bool(args.trace_edges)

    algo = args.algo.lower()
    if algo == "bfs":
        res = bfs(problem, max_expansions=args.max_expansions, trace=want_trace, trace_edges=trace_edges)
    elif algo == "dfs":
        res = dfs(problem, max_expansions=args.max_expansions, trace=want_trace, trace_edges=trace_edges)
    elif algo == "ucs":
        res = ucs(problem, max_expansions=args.max_expansions, trace=want_trace, trace_edges=trace_edges)
    elif algo in ("astar", "a*", "a_star"):
        res = astar(
            problem,
            heuristic=problem.admissible_heuristic,
            max_expansions=args.max_expansions,
            trace=want_trace,
            trace_edges=trace_edges,
        )
    else:
        raise ValueError(f"Unknown algo: {args.algo}")

    _print_search_result(res)

    if args.show_plan:
        print("\nplan:")
        for i, (s, a) in enumerate(zip(res.states, [None] + res.actions), start=0):
            if a is None:
                print(f"  {i:>3}: start @ {s}")
            else:
                print(f"  {i:>3}: {a:<5} -> {s}")

    if args.dot_out:
        write_search_dot(
            res,
            args.dot_out,
            title=f"tram/{algo} N={problem.N} walk={costs.walk} tram={costs.tram}",
            max_nodes=args.dot_max_nodes,
        )
        print(f"\nwrote DOT: {args.dot_out}")

    if args.trace_out:
        if res.trace is None:
            raise RuntimeError("Internal: wanted trace but none was captured")
        write_search_trace_jsonl(res.trace, args.trace_out, max_edges=args.trace_max_edges)
        print(f"wrote trace JSONL: {args.trace_out}")

    if args.html_out:
        write_search_trace_html(
            res,
            args.html_out,
            title=f"tram/{algo} N={problem.N} walk={costs.walk} tram={costs.tram}",
            max_nodes=args.html_max_nodes,
            max_edges=args.html_max_edges,
        )
        print(f"wrote HTML viz: {args.html_out}")

    return 0


def _cmd_run_tram_mdp(args: argparse.Namespace) -> int:
    costs = TramCosts(walk=float(args.walk_cost), tram=float(args.tram_cost))
    mdp = TransportationMDP(int(args.N), fail_prob=float(args.fail_prob), costs=costs)

    def render(it, V, pi, delta):
        print(f"\niter={it} delta={delta:.3e}")
        print(render_tram_grid(V, pi, N=int(args.N)))

    render_fn = render if args.render_every > 0 else None

    algo = args.algo.lower()
    if algo in ("value", "value_iterationation", "vi"):
        res = value_iterationation(
            mdp,
            epsilon=float(args.epsilon),
            max_iters=int(args.max_iters),
            render_every=int(args.render_every),
            render_fn=render_fn,
        )
    elif algo in ("policy", "policy_iteration", "pi"):
        res = policy_iteration(
            mdp,
            max_iters=int(args.max_iters),
            eval_epsilon=float(args.eval_epsilon),
            eval_max_iters=int(args.eval_max_iters),
            render_every=int(args.render_every),
            render_fn=render_fn,
        )
    else:
        raise ValueError(f"Unknown MDP algo: {args.algo}")

    print(f"\nfinal: iters={res.iterations} delta={res.delta:.3e}")
    print(render_tram_grid(res.V, res.policy, N=int(args.N)))
    return 0


def _cmd_bench_tram(args: argparse.Namespace) -> int:
    Ns = _parse_int_list(args.Ns)
    algos: Sequence[str] = args.algo

    rows = benchmark_tram_search(
        Ns,
        walk_cost=args.walk_cost,
        tram_cost=args.tram_cost,
        algos=algos,
        repeats=args.repeats,
        max_expansions=args.max_expansions,
    )

    out_fp = sys.stdout if args.out == "-" else open(args.out, "w", encoding="utf-8")
    try:
        fmt = args.format.lower()
        if fmt == "jsonl":
            to_jsonl(rows, out_fp)
        elif fmt == "csv":
            to_csv(rows, out_fp)
        else:
            raise ValueError(f"Unknown format: {args.format}")
    finally:
        if out_fp is not sys.stdout:
            out_fp.close()

    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ai_toolkit", description="Tiny classic AI toolbox (search/MDP/games/optim/dp)")
    sub = p.add_subparsers(dest="cmd", required=True)

    # run
    run = sub.add_parser("run", help="Run an algorithm on a domain")
    run_sub = run.add_subparsers(dest="domain", required=True)

    # run: tram search
    run_tram = run_sub.add_parser("tram", help="Run search on the walk-vs-tram domain")
    run_tram.add_argument("--algo", default="astar", choices=["bfs", "dfs", "ucs", "astar"], help="search algorithm")
    run_tram.add_argument("--N", type=int, required=True, help="goal state (start is 1)")
    run_tram.add_argument("--walk-cost", type=float, default=1.0)
    run_tram.add_argument("--tram-cost", type=float, default=2.0)
    run_tram.add_argument("--max-expansions", type=int, default=10_000_000)
    run_tram.add_argument("--show-plan", action="store_true", help="print the plan states/actions")

    # visualization outputs (v1.3)
    run_tram.add_argument("--dot-out", default=None, help="write a Graphviz DOT of the explored parent tree")
    run_tram.add_argument("--dot-max-nodes", type=int, default=2000, help="cap DOT nodes to avoid giant files")
    run_tram.add_argument("--trace-out", default=None, help="write a JSONL trace (expands + optional edges)")
    run_tram.add_argument("--trace-edges", action="store_true", help="include generated edges in JSONL trace")
    run_tram.add_argument("--trace-max-edges", type=int, default=1_000_000, help="cap edge events")

    # HTML visualizer output (v1.5)
    run_tram.add_argument("--html-out", default=None, help="write a self-contained HTML visualizer")
    run_tram.add_argument("--html-max-nodes", type=int, default=5000, help="cap HTML nodes")
    run_tram.add_argument("--html-max-edges", type=int, default=20000, help="cap HTML edges")

    run_tram.set_defaults(_handler=_cmd_run_tram)

    # run: tram MDP (v1.4)
    run_mdp = run_sub.add_parser("tram-mdp", help="Run value/policy iteration on the tram MDP")
    run_mdp.add_argument("--algo", default="value", choices=["value", "policy"], help="MDP algorithm")
    run_mdp.add_argument("--N", type=int, required=True)
    run_mdp.add_argument("--fail-prob", type=float, default=0.9)
    run_mdp.add_argument("--walk-cost", type=float, default=1.0)
    run_mdp.add_argument("--tram-cost", type=float, default=2.0)

    run_mdp.add_argument("--max-iters", type=int, default=100_000, help="max outer iterations")

    # value iteration
    run_mdp.add_argument("--epsilon", type=float, default=1e-10, help="value-iteration convergence threshold")

    # policy iteration
    run_mdp.add_argument("--eval-epsilon", type=float, default=1e-10, help="policy-eval convergence threshold")
    run_mdp.add_argument("--eval-max-iters", type=int, default=100_000, help="max policy-eval sweeps")

    # rendering
    run_mdp.add_argument("--render-every", type=int, default=0, help="render a stable table every k iterations")

    run_mdp.set_defaults(_handler=_cmd_run_tram_mdp)

    # bench tram
    bench = sub.add_parser("bench", help="Benchmark algorithms")
    bench_sub = bench.add_subparsers(dest="domain", required=True)

    bench_tram = bench_sub.add_parser("tram", help="Benchmark search on the walk-vs-tram domain")
    bench_tram.add_argument("--Ns", required=True, help='e.g. "10,20,30" or "5:200:5"')
    bench_tram.add_argument("--algo", nargs="+", default=["ucs", "astar"], help="algorithms to benchmark")
    bench_tram.add_argument("--repeats", type=int, default=1)
    bench_tram.add_argument("--walk-cost", type=float, default=1.0)
    bench_tram.add_argument("--tram-cost", type=float, default=2.0)
    bench_tram.add_argument("--max-expansions", type=int, default=10_000_000)
    bench_tram.add_argument("--format", default="jsonl", choices=["jsonl", "csv"])
    bench_tram.add_argument("--out", default="-", help="output path or '-' for stdout")
    bench_tram.set_defaults(_handler=_cmd_bench_tram)

    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return int(args._handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
