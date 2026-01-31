# ai_refactor_toolkit

A small, reusable toolbox of classic AI algorithms refactored from your scripts.

## What’s inside

- `ai_toolkit/search.py`: BFS / DFS / Uniform-Cost Search (Dijkstra) / A*
- `ai_toolkit/mdp.py`: Value Iteration
- `ai_toolkit/games.py`: Minimax + Alpha-Beta pruning
- `ai_toolkit/ml.py`: Sparse features + perceptron trainer
- `ai_toolkit/optim.py`: Batch GD + Mini-batch SGD for linear regression (NumPy)
- `ai_toolkit/dp.py`: Edit distance (top-down + bottom-up)
- `ai_toolkit/domains/tram.py`: “walk vs tram” domain + DP solver + structured perceptron helper
- `ai_toolkit/examples/*`: runnable demos

## Run

From this folder:

```bash
python -m unittest discover -s tests
python run_tests.py
python -m ai_toolkit.examples.tram_demo
python -m ai_toolkit.examples.tram_mdp_demo
python -m ai_toolkit.examples.halving_game_demo
python -m ai_toolkit.examples.linear_regression_demo
python -m ai_toolkit.examples.edit_distance_demo
python -m ai_toolkit.examples.feature_extractor_demo
```

## CLI

The package is runnable as a module:

```bash
python -m ai_toolkit run tram --algo astar --N 50 --walk-cost 1 --tram-cost 2 --show-plan
python -m ai_toolkit run tram --algo ucs   --N 100
```

Search runs now report instrumentation:

- expanded states
- generated successor edges
- reopens (UCS/A* improvements to previously seen states)
- max frontier size
- runtime (seconds)

## Benchmark harness

Benchmarks are designed to be easy to dump into spreadsheets.

```bash
# JSON Lines
python -m ai_toolkit bench tram --Ns "10,20,30" --algo ucs astar --repeats 3 --format jsonl --out results.jsonl

# CSV
python -m ai_toolkit bench tram --Ns "5:200:5" --algo ucs astar bfs --format csv --out results.csv
```

Each row includes the DP-optimal cost and the algorithm's optimality gap as a sanity check.

## Packaging

Editable install for development:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -e .
```

## Notes

- The original `gradientDescentStochastic.py` generated 500k points and looped in Python.
  Here we keep the idea but implement vectorized NumPy + mini-batches for speed.

- The original `tram-MDPs.py` paused every iteration. The demo prints the final policy
  after convergence.


## Legacy

The original scripts you uploaded are preserved under `legacy/` for reference.
