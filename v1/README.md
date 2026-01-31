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
python -m unittest
python -m ai_toolkit.examples.tram_demo
python -m ai_toolkit.examples.tram_mdp_demo
python -m ai_toolkit.examples.halving_game_demo
python -m ai_toolkit.examples.linear_regression_demo
python -m ai_toolkit.examples.edit_distance_demo
python -m ai_toolkit.examples.feature_extractor_demo
```

## Notes

- The original `gradientDescentStochastic.py` generated 500k points and looped in Python.
  Here we keep the idea but implement vectorized NumPy + mini-batches for speed.

- The original `tram-MDPs.py` paused every iteration. The demo prints the final policy
  after convergence.


## Legacy

The original scripts you uploaded are preserved under `legacy/` for reference.
