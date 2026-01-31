"""Compatibility entrypoint.

`python -m ai_toolkit ...` delegates to `ai_toolkit.cli`.
"""

from __future__ import annotations

from .cli.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main())
