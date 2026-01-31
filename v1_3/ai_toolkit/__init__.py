"""
ai_toolkit: a small, reusable grab-bag of classic AI algorithms.

Modules:
  - search: BFS/DFS/UCS/A*
  - mdp: Value iteration + helpers
  - games: Minimax + alpha-beta pruning
  - optim: Batch GD + SGD for linear regression
  - dp: Edit distance (top-down + bottom-up)
  - domains.tram: the walk/tram toy domain used by the included examples
"""
__version__ = "0.1.3"

__all__ = ["search", "mdp", "games", "optim", "dp", "bench"]
