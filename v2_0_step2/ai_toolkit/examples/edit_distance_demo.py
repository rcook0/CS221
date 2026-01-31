from __future__ import annotations

from ai_toolkit.dp import edit_distance_topdown, edit_distance_bottomup


def main() -> None:
    s = "a cat!" * 10
    t = "the cats!" * 10
    print("topdown:", edit_distance_topdown(s, t))
    print("bottomup:", edit_distance_bottomup(s, t))


if __name__ == "__main__":
    main()
