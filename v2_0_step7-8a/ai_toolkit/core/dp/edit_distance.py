from __future__ import annotations

from functools import lru_cache
from typing import List


def edit_distance_topdown(s: str, t: str) -> int:
    @lru_cache(maxsize=None)
    def rec(m: int, n: int) -> int:
        if m == 0:
            return n
        if n == 0:
            return m
        if s[m - 1] == t[n - 1]:
            return rec(m - 1, n - 1)
        return 1 + min(
            rec(m - 1, n - 1),
            rec(m - 1, n),
            rec(m, n - 1),
        )
    return rec(len(s), len(t))


def edit_distance_bottomup(s: str, t: str) -> int:
    m, n = len(s), len(t)
    dp: List[List[int]] = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        si = s[i - 1]
        for j in range(1, n + 1):
            if si == t[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]
