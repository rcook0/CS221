"""ai_toolkit: a small, reusable grab-bag of classic AI algorithms.

v2.0 introduces an internal-library layout split into:
  - ai_toolkit.core      (algorithms + shared structures)
  - ai_toolkit.domains   (reference problems)
  - ai_toolkit.viz       (offline visualization helpers)
  - ai_toolkit.cli       (command-line interface)
  - ai_toolkit.experimental (sandbox; not stable)

Compatibility re-exports remain at the package root (ai_toolkit.search, ai_toolkit.mdp, ...).
"""

from .__version__ import __version__

__all__ = [
    "core",
    "domains",
    "viz",
    "cli",
    "experimental",
    # compat
    "search",
    "mdp",
    "games",
    "optim",
    "dp",
    "ml",
    "bench",
    "structures",
]
