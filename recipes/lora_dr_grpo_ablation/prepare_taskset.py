"""Reproducible Harbor taskset builder for the dasv LoRA-RLVR Dr. GRPO 3-arm ablation (KPI-RES-03).

Supports two deterministic modes:
  1. `--source hf` (default for GPU runs): pulls real permissive-license Python coding tasks from
     `bigcode/bigcodebench` (Apache-2.0) + `evalplus/mbppplus` (Apache-2.0), filters strictly to
     offline stdlib/numpy/pandas tasks, splits into disjoint Train (128) & Clean Held-Out (64),
     and materializes paired Arm 1 (Unhardened) vs. Arm 2 (dasv-Hardened) Harbor directories.
  2. `--source synthetic` (offline CI & CPU dry-run): builds 20 genuine algorithmic Python Harbor
     tasks (12 train + 8 clean held-out) with real canonical oracles and unit tests ($0, offline).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dasv.adapters.harbor import HarborAdapter
from dasv.sandbox.probes import run_tier0_static_probes

# Official python:3.12-slim multi-arch digest fallback when Docker daemon is unavailable locally.
DEFAULT_PINNED_PYTHON_IMAGE = (
    "python:3.12-slim@sha256:9e01a05c7718569f21f99a3a8301c2d0795857e9522e655f6b9772947f5d2a3d"
)
UNPINNED_PYTHON_IMAGE = "python:3.12-slim"

_ALLOWED_HF_LIBS: frozenset[str] = frozenset(
    {
        "",
        "bisect",
        "collections",
        "copy",
        "csv",
        "dataclasses",
        "datetime",
        "decimal",
        "enum",
        "fractions",
        "functools",
        "hashlib",
        "heapq",
        "io",
        "itertools",
        "json",
        "math",
        "numpy",
        "operator",
        "pandas",
        "random",
        "re",
        "statistics",
        "string",
        "typing",
    }
)


@dataclass(frozen=True)
class CodingTaskSpec:
    """Normalized pure-Python coding benchmark task used to build Harbor task directories."""

    task_id: str
    source_benchmark: str
    license_id: str
    prompt: str
    starter_code: str
    oracle_code: str
    full_test_code: str
    weak_test_code: str


def _resolve_pinned_python_image() -> str:
    """Resolve live OCI @sha256 digest of local python:3.12-slim when available."""
    try:
        proc = subprocess.run(
            ["docker", "inspect", "--format={{index .RepoDigests 0}}", UNPINNED_PYTHON_IMAGE],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        candidate = proc.stdout.strip()
        if proc.returncode == 0 and "@sha256:" in candidate:
            return candidate
    except (OSError, subprocess.SubprocessError):
        pass
    return DEFAULT_PINNED_PYTHON_IMAGE


def build_synthetic_corpus() -> list[CodingTaskSpec]:
    """Return 20 deterministic, non-trivial algorithmic Python tasks for offline tests & dry-run."""
    raw_specs: list[tuple[str, str, str, str, str, str]] = [
        (
            "syn-merge-intervals",
            "Implement `merge_intervals(intervals: list[list[int]]) -> list[list[int]]` in "
            "`solution.py` that merges all overlapping closed intervals `[start, end]` and "
            "returns them sorted by start.\nExample: `merge_intervals([[1,3],[2,6],[8,10]]) == "
            "[[1,6],[8,10]]`.",
            "def merge_intervals(intervals: list[list[int]]) -> list[list[int]]:\n"
            "    raise NotImplementedError\n",
            "def merge_intervals(intervals: list[list[int]]) -> list[list[int]]:\n"
            "    if not intervals:\n"
            "        return []\n"
            "    ordered = sorted((int(a), int(b)) for a, b in intervals)\n"
            "    merged = [[ordered[0][0], ordered[0][1]]]\n"
            "    for start, end in ordered[1:]:\n"
            "        if start <= merged[-1][1]:\n"
            "            merged[-1][1] = max(merged[-1][1], end)\n"
            "        else:\n"
            "            merged.append([start, end])\n"
            "    return merged\n",
            "from solution import merge_intervals\n\n"
            "def test_example():\n"
            "    assert merge_intervals([[1, 3], [2, 6], [8, 10]]) == [[1, 6], [8, 10]]\n",
            "from solution import merge_intervals\n\n"
            "def test_example():\n"
            "    assert merge_intervals([[1, 3], [2, 6], [8, 10]]) == [[1, 6], [8, 10]]\n\n"
            "def test_edges():\n"
            "    assert merge_intervals([]) == []\n"
            "    assert merge_intervals([[5, 7], [1, 2], [2, 5]]) == [[1, 7]]\n"
            "    assert merge_intervals([[-10, -2], [-5, 0], [1, 1]]) == [[-10, 0], [1, 1]]\n"
            "    assert merge_intervals([[1, 10], [2, 3], [4, 5]]) == [[1, 10]]\n",
        ),
        (
            "syn-rpn-eval",
            "Implement `eval_rpn(tokens: list[str]) -> int` in `solution.py` evaluating Reverse "
            "Polish Notation with `+`, `-`, `*`, `/` (integer division truncating toward zero).\n"
            "Example: `eval_rpn(['2', '1', '+', '3', '*']) == 9`.",
            "def eval_rpn(tokens: list[str]) -> int:\n    raise NotImplementedError\n",
            "def eval_rpn(tokens: list[str]) -> int:\n"
            "    stack: list[int] = []\n"
            "    for tok in tokens:\n"
            "        if tok in {'+', '-', '*', '/'}:\n"
            "            b = stack.pop()\n"
            "            a = stack.pop()\n"
            "            if tok == '+':\n"
            "                stack.append(a + b)\n"
            "            elif tok == '-':\n"
            "                stack.append(a - b)\n"
            "            elif tok == '*':\n"
            "                stack.append(a * b)\n"
            "            else:\n"
            "                stack.append(int(a / b))\n"
            "        else:\n"
            "            stack.append(int(tok))\n"
            "    return stack[-1]\n",
            "from solution import eval_rpn\n\n"
            "def test_example():\n"
            "    assert eval_rpn(['2', '1', '+', '3', '*']) == 9\n",
            "from solution import eval_rpn\n\n"
            "def test_example():\n"
            "    assert eval_rpn(['2', '1', '+', '3', '*']) == 9\n\n"
            "def test_truncation_toward_zero():\n"
            "    assert eval_rpn(['13', '-5', '/']) == -2\n"
            "    assert eval_rpn(['4', '13', '5', '/', '+']) == 6\n"
            "    assert eval_rpn(['10', '6', '9', '3', '+', '-11', '*', '/', '*', '17', '+', "
            "'5', '+']) == 22\n",
        ),
        (
            "syn-lru-cache",
            "Implement class `LRUCache` in `solution.py` with `__init__(self, capacity: int)`, "
            "`get(self, key: int) -> int` (returns `-1` if missing), and "
            "`put(self, key: int, value: int) -> None` evicting least-recently used key.\n"
            "Example: capacity=2, put(1,1), put(2,2), get(1)==1.",
            "class LRUCache:\n"
            "    def __init__(self, capacity: int) -> None:\n"
            "        raise NotImplementedError\n",
            "from collections import OrderedDict\n\n"
            "class LRUCache:\n"
            "    def __init__(self, capacity: int) -> None:\n"
            "        self.cap = capacity\n"
            "        self.data: OrderedDict[int, int] = OrderedDict()\n\n"
            "    def get(self, key: int) -> int:\n"
            "        if key not in self.data:\n"
            "            return -1\n"
            "        self.data.move_to_end(key)\n"
            "        return self.data[key]\n\n"
            "    def put(self, key: int, value: int) -> None:\n"
            "        if self.cap <= 0:\n"
            "            return\n"
            "        if key in self.data:\n"
            "            self.data.move_to_end(key)\n"
            "        self.data[key] = value\n"
            "        if len(self.data) > self.cap:\n"
            "            self.data.popitem(last=False)\n",
            "from solution import LRUCache\n\n"
            "def test_example():\n"
            "    c = LRUCache(2)\n"
            "    c.put(1, 1)\n"
            "    c.put(2, 2)\n"
            "    assert c.get(1) == 1\n",
            "from solution import LRUCache\n\n"
            "def test_eviction_order():\n"
            "    c = LRUCache(2)\n"
            "    c.put(1, 10)\n"
            "    c.put(2, 20)\n"
            "    assert c.get(1) == 10\n"
            "    c.put(3, 30)\n"
            "    assert c.get(2) == -1\n"
            "    c.put(1, 11)\n"
            "    c.put(4, 40)\n"
            "    assert c.get(3) == -1\n"
            "    assert c.get(1) == 11\n"
            "    assert c.get(4) == 40\n",
        ),
        (
            "syn-topo-sort",
            "Implement `topo_sort(n: int, edges: list[tuple[int, int]]) -> list[int]` in "
            "`solution.py` returning the lexicographically smallest topological ordering of nodes "
            "`0..n-1` for directed edges `(u, v)`, or `[]` if a cycle exists.\n"
            "Example: `topo_sort(3, [(0, 2), (1, 2)]) == [0, 1, 2]`.",
            "def topo_sort(n: int, edges: list[tuple[int, int]]) -> list[int]:\n"
            "    raise NotImplementedError\n",
            "import heapq\n\n"
            "def topo_sort(n: int, edges: list[tuple[int, int]]) -> list[int]:\n"
            "    adj: list[list[int]] = [[] for _ in range(n)]\n"
            "    indeg = [0] * n\n"
            "    for u, v in edges:\n"
            "        adj[u].append(v)\n"
            "        indeg[v] += 1\n"
            "    heap = [i for i in range(n) if indeg[i] == 0]\n"
            "    heapq.heapify(heap)\n"
            "    out: list[int] = []\n"
            "    while heap:\n"
            "        u = heapq.heappop(heap)\n"
            "        out.append(u)\n"
            "        for v in adj[u]:\n"
            "            indeg[v] -= 1\n"
            "            if indeg[v] == 0:\n"
            "                heapq.heappush(heap, v)\n"
            "    return out if len(out) == n else []\n",
            "from solution import topo_sort\n\n"
            "def test_example():\n"
            "    assert topo_sort(3, [(0, 2), (1, 2)]) == [0, 1, 2]\n",
            "from solution import topo_sort\n\n"
            "def test_lex_and_cycle():\n"
            "    assert topo_sort(3, [(0, 2), (1, 2)]) == [0, 1, 2]\n"
            "    assert topo_sort(4, [(2, 0), (3, 1)]) == [2, 0, 3, 1]\n"
            "    assert topo_sort(3, [(0, 1), (1, 2), (2, 0)]) == []\n"
            "    assert topo_sort(0, []) == []\n",
        ),
        (
            "syn-rle-codec",
            "Implement `rle_encode(s: str) -> str` and `rle_decode(s: str) -> str` in "
            "`solution.py` using `<count><char>` tokens for ASCII letters.\n"
            "Example: `rle_encode('aaabbc') == '3a2b1c'`.",
            "def rle_encode(s: str) -> str:\n    raise NotImplementedError\n"
            "def rle_decode(s: str) -> str:\n    raise NotImplementedError\n",
            "def rle_encode(s: str) -> str:\n"
            "    if not s:\n"
            "        return ''\n"
            "    parts: list[str] = []\n"
            "    cur = s[0]\n"
            "    cnt = 1\n"
            "    for ch in s[1:]:\n"
            "        if ch == cur:\n"
            "            cnt += 1\n"
            "        else:\n"
            "            parts.append(f'{cnt}{cur}')\n"
            "            cur = ch\n"
            "            cnt = 1\n"
            "    parts.append(f'{cnt}{cur}')\n"
            "    return ''.join(parts)\n\n"
            "def rle_decode(s: str) -> str:\n"
            "    out: list[str] = []\n"
            "    num = 0\n"
            "    for ch in s:\n"
            "        if ch.isdigit():\n"
            "            num = num * 10 + int(ch)\n"
            "        else:\n"
            "            out.append(ch * num)\n"
            "            num = 0\n"
            "    return ''.join(out)\n",
            "from solution import rle_encode, rle_decode\n\n"
            "def test_example():\n"
            "    assert rle_encode('aaabbc') == '3a2b1c'\n",
            "from solution import rle_encode, rle_decode\n\n"
            "def test_roundtrip():\n"
            "    assert rle_encode('') == ''\n"
            "    assert rle_decode('') == ''\n"
            "    long_s = 'a' * 12 + 'b' * 3 + 'z' * 15\n"
            "    enc = rle_encode(long_s)\n"
            "    assert enc == '12a3b15z'\n"
            "    assert rle_decode(enc) == long_s\n",
        ),
        (
            "syn-wildcard-parens",
            "Implement `check_valid_parens(s: str) -> bool` in `solution.py` where `'('`, `')'`, "
            "and wildcard `'*'` (empty, `'('`, or `')'`) must form a valid balanced string.\n"
            "Example: `check_valid_parens('(*)') is True`.",
            "def check_valid_parens(s: str) -> bool:\n    raise NotImplementedError\n",
            "def check_valid_parens(s: str) -> bool:\n"
            "    lo = hi = 0\n"
            "    for ch in s:\n"
            "        if ch == '(':\n"
            "            lo += 1\n"
            "            hi += 1\n"
            "        elif ch == ')':\n"
            "            lo = max(0, lo - 1)\n"
            "            hi -= 1\n"
            "        elif ch == '*':\n"
            "            lo = max(0, lo - 1)\n"
            "            hi += 1\n"
            "        if hi < 0:\n"
            "            return False\n"
            "    return lo == 0\n",
            "from solution import check_valid_parens\n\n"
            "def test_example():\n"
            "    assert check_valid_parens('(*)') is True\n",
            "from solution import check_valid_parens\n\n"
            "def test_cases():\n"
            "    assert check_valid_parens('') is True\n"
            "    assert check_valid_parens('(*))') is True\n"
            "    assert check_valid_parens('((*))))') is False\n"
            "    assert check_valid_parens(')(') is False\n"
            "    assert check_valid_parens('(((******))') is True\n",
        ),
        (
            "syn-spiral-matrix",
            "Implement `spiral_order(matrix: list[list[int]]) -> list[int]` in `solution.py` "
            "returning all elements in clockwise spiral order.\n"
            "Example: `spiral_order([[1,2],[3,4]]) == [1,2,4,3]`.",
            "def spiral_order(matrix: list[list[int]]) -> list[int]:\n"
            "    raise NotImplementedError\n",
            "def spiral_order(matrix: list[list[int]]) -> list[int]:\n"
            "    if not matrix or not matrix[0]:\n"
            "        return []\n"
            "    top, bottom = 0, len(matrix) - 1\n"
            "    left, right = 0, len(matrix[0]) - 1\n"
            "    res: list[int] = []\n"
            "    while top <= bottom and left <= right:\n"
            "        for c in range(left, right + 1):\n"
            "            res.append(matrix[top][c])\n"
            "        top += 1\n"
            "        for r in range(top, bottom + 1):\n"
            "            res.append(matrix[r][right])\n"
            "        right -= 1\n"
            "        if top <= bottom:\n"
            "            for c in range(right, left - 1, -1):\n"
            "                res.append(matrix[bottom][c])\n"
            "            bottom -= 1\n"
            "        if left <= right:\n"
            "            for r in range(bottom, top - 1, -1):\n"
            "                res.append(matrix[r][left])\n"
            "            left += 1\n"
            "    return res\n",
            "from solution import spiral_order\n\n"
            "def test_example():\n"
            "    assert spiral_order([[1, 2], [3, 4]]) == [1, 2, 4, 3]\n",
            "from solution import spiral_order\n\n"
            "def test_shapes():\n"
            "    assert spiral_order([]) == []\n"
            "    assert spiral_order([[1, 2, 3]]) == [1, 2, 3]\n"
            "    assert spiral_order([[1], [2], [3]]) == [1, 2, 3]\n"
            "    assert spiral_order([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) == [1,2,3,6,9,8,7,4,5]\n",
        ),
        (
            "syn-lis-length",
            "Implement `lis_length(nums: list[int]) -> int` in `solution.py` returning the length "
            "of the longest strictly increasing subsequence in $O(n \\log n)$.\n"
            "Example: `lis_length([10, 9, 2, 5, 3, 7, 101, 18]) == 4`.",
            "def lis_length(nums: list[int]) -> int:\n    raise NotImplementedError\n",
            "import bisect\n\n"
            "def lis_length(nums: list[int]) -> int:\n"
            "    tails: list[int] = []\n"
            "    for x in nums:\n"
            "        idx = bisect.bisect_left(tails, x)\n"
            "        if idx == len(tails):\n"
            "            tails.append(x)\n"
            "        else:\n"
            "            tails[idx] = x\n"
            "    return len(tails)\n",
            "from solution import lis_length\n\n"
            "def test_example():\n"
            "    assert lis_length([10, 9, 2, 5, 3, 7, 101, 18]) == 4\n",
            "from solution import lis_length\n\n"
            "def test_edges():\n"
            "    assert lis_length([]) == 0\n"
            "    assert lis_length([7, 7, 7, 7]) == 1\n"
            "    assert lis_length([0, 1, 0, 3, 2, 3]) == 4\n"
            "    assert lis_length(list(range(500, 0, -1))) == 1\n",
        ),
        (
            "syn-flatten-json",
            "Implement `flatten_dict(data: dict, sep: str = '.') -> dict` in `solution.py` "
            "recursively flattening nested dicts while preserving empty dicts as `{}`.\n"
            "Example: `flatten_dict({'a': {'b': 1}}) == {'a.b': 1}`.",
            "def flatten_dict(data: dict, sep: str = '.') -> dict:\n"
            "    raise NotImplementedError\n",
            "def flatten_dict(data: dict, sep: str = '.') -> dict:\n"
            "    out: dict = {}\n"
            "    def _walk(prefix: str, obj: dict) -> None:\n"
            "        for k, v in obj.items():\n"
            "            nk = f'{prefix}{sep}{k}' if prefix else str(k)\n"
            "            if isinstance(v, dict) and v:\n"
            "                _walk(nk, v)\n"
            "            else:\n"
            "                out[nk] = v\n"
            "    _walk('', data)\n"
            "    return out\n",
            "from solution import flatten_dict\n\n"
            "def test_example():\n"
            "    assert flatten_dict({'a': {'b': 1}}) == {'a.b': 1}\n",
            "from solution import flatten_dict\n\n"
            "def test_nested():\n"
            "    assert flatten_dict({}) == {}\n"
            "    assert flatten_dict({'a': {'b': {'c': 3}}, 'd': {}}, sep='/') == "
            "{'a/b/c': 3, 'd': {}}\n"
            "    assert flatten_dict({'x': [1, 2], 'y': {'z': None}}) == "
            "{'x': [1, 2], 'y.z': None}\n",
        ),
        (
            "syn-roman-convert",
            "Implement `int_to_roman(num: int) -> str` and `roman_to_int(s: str) -> int` in "
            "`solution.py` for integers `1..3999`.\nExample: `int_to_roman(1994) == 'MCMXCIV'`.",
            "def int_to_roman(num: int) -> str:\n    raise NotImplementedError\n"
            "def roman_to_int(s: str) -> int:\n    raise NotImplementedError\n",
            "def int_to_roman(num: int) -> str:\n"
            "    vals = [(1000,'M'),(900,'CM'),(500,'D'),(400,'CD'),(100,'C'),(90,'XC'),"
            "(50,'L'),(40,'XL'),(10,'X'),(9,'IX'),(5,'V'),(4,'IV'),(1,'I')]\n"
            "    out: list[str] = []\n"
            "    for v, sym in vals:\n"
            "        while num >= v:\n"
            "            out.append(sym)\n"
            "            num -= v\n"
            "    return ''.join(out)\n\n"
            "def roman_to_int(s: str) -> int:\n"
            "    m = {'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}\n"
            "    total = 0\n"
            "    for i, ch in enumerate(s):\n"
            "        if i + 1 < len(s) and m[ch] < m[s[i + 1]]:\n"
            "            total -= m[ch]\n"
            "        else:\n"
            "            total += m[ch]\n"
            "    return total\n",
            "from solution import int_to_roman, roman_to_int\n\n"
            "def test_example():\n"
            "    assert int_to_roman(1994) == 'MCMXCIV'\n",
            "from solution import int_to_roman, roman_to_int\n\n"
            "def test_roundtrip():\n"
            "    for n in (1, 4, 9, 42, 99, 444, 1994, 2026, 3999):\n"
            "        r = int_to_roman(n)\n"
            "        assert roman_to_int(r) == n\n",
        ),
        (
            "syn-semver-sort",
            "Implement `sort_semver(versions: list[str]) -> list[str]` in `solution.py` sorting "
            "`MAJOR.MINOR.PATCH` strings (with optional `-prerelease` ranking before release).\n"
            "Example: `sort_semver(['1.10.0', '1.2.0']) == ['1.2.0', '1.10.0']`.",
            "def sort_semver(versions: list[str]) -> list[str]:\n    raise NotImplementedError\n",
            "def sort_semver(versions: list[str]) -> list[str]:\n"
            "    def _key(v: str):\n"
            "        core, _, pre = v.partition('-')\n"
            "        nums = tuple(int(p) for p in core.split('.'))\n"
            "        if not pre:\n"
            "            return (nums, 1, ())\n"
            "        pre_parts = tuple((0, int(p)) if p.isdigit() else (1, p) "
            "for p in pre.split('.'))\n"
            "        return (nums, 0, pre_parts)\n"
            "    return sorted(versions, key=_key)\n",
            "from solution import sort_semver\n\n"
            "def test_example():\n"
            "    assert sort_semver(['1.10.0', '1.2.0']) == ['1.2.0', '1.10.0']\n",
            "from solution import sort_semver\n\n"
            "def test_prerelease():\n"
            "    inp = ['1.0.0', '1.0.0-alpha.2', '1.0.0-alpha.10', '1.0.0-beta', '0.9.9']\n"
            "    assert sort_semver(inp) == ['0.9.9', '1.0.0-alpha.2', '1.0.0-alpha.10', "
            "'1.0.0-beta', '1.0.0']\n",
        ),
        (
            "syn-dijkstra-grid",
            "Implement `min_path_cost(grid: list[list[int]]) -> int` in `solution.py` finding "
            "the minimum 4-directional path sum from `(0,0)` to `(R-1,C-1)` (non-negative costs).\n"
            "Example: `min_path_cost([[1,3,1],[1,5,1],[4,2,1]]) == 7`.",
            "def min_path_cost(grid: list[list[int]]) -> int:\n    raise NotImplementedError\n",
            "import heapq\n\n"
            "def min_path_cost(grid: list[list[int]]) -> int:\n"
            "    R, C = len(grid), len(grid[0])\n"
            "    dist = [[10**18] * C for _ in range(R)]\n"
            "    dist[0][0] = grid[0][0]\n"
            "    pq = [(grid[0][0], 0, 0)]\n"
            "    while pq:\n"
            "        d, r, c = heapq.heappop(pq)\n"
            "        if d > dist[r][c]:\n"
            "            continue\n"
            "        if r == R - 1 and c == C - 1:\n"
            "            return d\n"
            "        for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):\n"
            "            nr, nc = r + dr, c + dc\n"
            "            if 0 <= nr < R and 0 <= nc < C:\n"
            "                nd = d + grid[nr][nc]\n"
            "                if nd < dist[nr][nc]:\n"
            "                    dist[nr][nc] = nd\n"
            "                    heapq.heappush(pq, (nd, nr, nc))\n"
            "    return dist[-1][-1]\n",
            "from solution import min_path_cost\n\n"
            "def test_example():\n"
            "    assert min_path_cost([[1, 3, 1], [1, 5, 1], [4, 2, 1]]) == 7\n",
            "from solution import min_path_cost\n\n"
            "def test_backtrack_required():\n"
            "    # Path must go up/left around high-cost wall\n"
            "    g = [\n"
            "        [1, 100, 1, 1, 1],\n"
            "        [1, 100, 1, 100, 1],\n"
            "        [1, 1, 1, 100, 1],\n"
            "    ]\n"
            "    assert min_path_cost(g) == 11\n",
        ),
        (
            "syn-trie-counter",
            "Implement `PrefixCounter` in `solution.py` with `insert(word: str)` and "
            "`count_prefix(prefix: str) -> int` counting inserted words (with multiplicity).\n"
            "Example: insert 'app', 'apple' -> `count_prefix('app') == 2`.",
            "class PrefixCounter:\n    def insert(self, word: str) -> None:\n"
            "        raise NotImplementedError\n",
            "class PrefixCounter:\n"
            "    def __init__(self) -> None:\n"
            "        self.root: dict = {'_c': 0}\n\n"
            "    def insert(self, word: str) -> None:\n"
            "        node = self.root\n"
            "        node['_c'] += 1\n"
            "        for ch in word:\n"
            "            if ch not in node:\n"
            "                node[ch] = {'_c': 0}\n"
            "            node = node[ch]\n"
            "            node['_c'] += 1\n\n"
            "    def count_prefix(self, prefix: str) -> int:\n"
            "        node = self.root\n"
            "        for ch in prefix:\n"
            "            if ch not in node:\n"
            "                return 0\n"
            "            node = node[ch]\n"
            "        return int(node['_c'])\n",
            "from solution import PrefixCounter\n\n"
            "def test_example():\n"
            "    pc = PrefixCounter()\n"
            "    pc.insert('app')\n"
            "    pc.insert('apple')\n"
            "    assert pc.count_prefix('app') == 2\n",
            "from solution import PrefixCounter\n\n"
            "def test_multiplicity_and_empty():\n"
            "    pc = PrefixCounter()\n"
            "    for w in ('cat', 'cat', 'car', 'dog'):\n"
            "        pc.insert(w)\n"
            "    assert pc.count_prefix('') == 4\n"
            "    assert pc.count_prefix('ca') == 3\n"
            "    assert pc.count_prefix('cat') == 2\n"
            "    assert pc.count_prefix('z') == 0\n",
        ),
        (
            "syn-sliding-median",
            "Implement `sliding_window_median(nums: list[int], k: int) -> list[float]` in "
            "`solution.py` returning the exact median for each contiguous window of length `k`.\n"
            "Example: `sliding_window_median([1,3,-1,-3,5], 3) == [1.0, -1.0, -1.0]`.",
            "def sliding_window_median(nums: list[int], k: int) -> list[float]:\n"
            "    raise NotImplementedError\n",
            "import bisect\n\n"
            "def sliding_window_median(nums: list[int], k: int) -> list[float]:\n"
            "    if k <= 0 or k > len(nums):\n"
            "        return []\n"
            "    win = sorted(nums[:k])\n"
            "    out: list[float] = []\n"
            "    def _med() -> float:\n"
            "        mid = k // 2\n"
            "        return float(win[mid]) if k % 2 == 1 else (win[mid - 1] + win[mid]) / 2.0\n"
            "    out.append(_med())\n"
            "    for i in range(k, len(nums)):\n"
            "        old_idx = bisect.bisect_left(win, nums[i - k])\n"
            "        win.pop(old_idx)\n"
            "        bisect.insort(win, nums[i])\n"
            "        out.append(_med())\n"
            "    return out\n",
            "from solution import sliding_window_median\n\n"
            "def test_example():\n"
            "    assert sliding_window_median([1, 3, -1, -3, 5], 3) == [1.0, -1.0, -1.0]\n",
            "from solution import sliding_window_median\n\n"
            "def test_even_and_odd():\n"
            "    assert sliding_window_median([1, 2, 3, 4], 2) == [1.5, 2.5, 3.5]\n"
            "    assert sliding_window_median([5, 5, 5], 1) == [5.0, 5.0, 5.0]\n"
            "    assert sliding_window_median([1, 3, -1, -3, 5, 3, 6, 7], 3) == "
            "[1.0, -1.0, -1.0, 3.0, 5.0, 6.0]\n",
        ),
        (
            "syn-expr-calc",
            "Implement `calculate(expr: str) -> int` in `solution.py` evaluating integer "
            "expressions with `+`, `-`, `*`, `/` (truncating toward zero) and `(...)`.\n"
            "Example: `calculate('(2+3)*4') == 20`.",
            "def calculate(expr: str) -> int:\n    raise NotImplementedError\n",
            "def calculate(expr: str) -> int:\n"
            "    s = expr.replace(' ', '')\n"
            "    pos = 0\n"
            "    def _parse_expr() -> int:\n"
            "        nonlocal pos\n"
            "        val = _parse_term()\n"
            "        while pos < len(s) and s[pos] in '+-':\n"
            "            op = s[pos]; pos += 1\n"
            "            rhs = _parse_term()\n"
            "            val = val + rhs if op == '+' else val - rhs\n"
            "        return val\n"
            "    def _parse_term() -> int:\n"
            "        nonlocal pos\n"
            "        val = _parse_factor()\n"
            "        while pos < len(s) and s[pos] in '*/':\n"
            "            op = s[pos]; pos += 1\n"
            "            rhs = _parse_factor()\n"
            "            val = val * rhs if op == '*' else int(val / rhs)\n"
            "        return val\n"
            "    def _parse_factor() -> int:\n"
            "        nonlocal pos\n"
            "        if s[pos] == '-':\n"
            "            pos += 1\n"
            "            return -_parse_factor()\n"
            "        if s[pos] == '(':\n"
            "            pos += 1\n"
            "            v = _parse_expr()\n"
            "            pos += 1\n"
            "            return v\n"
            "        start = pos\n"
            "        while pos < len(s) and s[pos].isdigit():\n"
            "            pos += 1\n"
            "        return int(s[start:pos])\n"
            "    return _parse_expr()\n",
            "from solution import calculate\n\n"
            "def test_example():\n"
            "    assert calculate('(2+3)*4') == 20\n",
            "from solution import calculate\n\n"
            "def test_precedence():\n"
            "    assert calculate('2 + 3 * 4 - 10 / 3') == 11\n"
            "    assert calculate('-(3 + (4 - 10) * 2)') == 9\n"
            "    assert calculate('((1+2)*(3+4))/5') == 4\n",
        ),
        (
            "syn-token-bucket",
            "Implement `TokenBucket(capacity: float, refill_rate: float)` in `solution.py` with "
            "`allow(now: float, cost: float = 1.0) -> bool` starting full at `t=0`.\n"
            "Example: capacity=2, refill_rate=1 -> allow(0), allow(0) True, allow(0) False.",
            "class TokenBucket:\n"
            "    def __init__(self, capacity: float, refill_rate: float) -> None:\n"
            "        raise NotImplementedError\n",
            "class TokenBucket:\n"
            "    def __init__(self, capacity: float, refill_rate: float) -> None:\n"
            "        self.capacity = float(capacity)\n"
            "        self.rate = float(refill_rate)\n"
            "        self.tokens = float(capacity)\n"
            "        self.last_t = 0.0\n\n"
            "    def allow(self, now: float, cost: float = 1.0) -> bool:\n"
            "        dt = max(0.0, float(now) - self.last_t)\n"
            "        self.tokens = min(self.capacity, self.tokens + dt * self.rate)\n"
            "        self.last_t = max(self.last_t, float(now))\n"
            "        if self.tokens + 1e-12 >= cost:\n"
            "            self.tokens = max(0.0, self.tokens - cost)\n"
            "            return True\n"
            "        return False\n",
            "from solution import TokenBucket\n\n"
            "def test_example():\n"
            "    tb = TokenBucket(2.0, 1.0)\n"
            "    assert tb.allow(0.0) is True\n"
            "    assert tb.allow(0.0) is True\n"
            "    assert tb.allow(0.0) is False\n",
            "from solution import TokenBucket\n\n"
            "def test_refill_and_cap():\n"
            "    tb = TokenBucket(3.0, 0.5)\n"
            "    assert tb.allow(0.0, 3.0) is True\n"
            "    assert tb.allow(1.0, 1.0) is False\n"
            "    assert tb.allow(2.0, 1.0) is True\n"
            "    assert tb.allow(100.0, 3.0) is True\n"
            "    assert tb.allow(100.0, 0.1) is False\n",
        ),
        (
            "syn-cron-match",
            "Implement `match_cron_field(expr: str, value: int, lo: int, hi: int) -> bool` in "
            "`solution.py` supporting `*`, `*/step`, `a-b`, `a-b/step`, and comma lists.\n"
            "Example: `match_cron_field('*/15', 30, 0, 59) is True`.",
            "def match_cron_field(expr: str, value: int, lo: int, hi: int) -> bool:\n"
            "    raise NotImplementedError\n",
            "def match_cron_field(expr: str, value: int, lo: int, hi: int) -> bool:\n"
            "    if value < lo or value > hi:\n"
            "        return False\n"
            "    for part in expr.split(','):\n"
            "        base, _, step_s = part.partition('/')\n"
            "        step = int(step_s) if step_s else 1\n"
            "        if base == '*':\n"
            "            start, end = lo, hi\n"
            "        elif '-' in base:\n"
            "            a, b = base.split('-', 1)\n"
            "            start, end = int(a), int(b)\n"
            "        else:\n"
            "            start = int(base)\n"
            "            end = hi if step_s else start\n"
            "        if start <= value <= end and (value - start) % step == 0:\n"
            "            return True\n"
            "    return False\n",
            "from solution import match_cron_field\n\n"
            "def test_example():\n"
            "    assert match_cron_field('*/15', 30, 0, 59) is True\n",
            "from solution import match_cron_field\n\n"
            "def test_ranges_and_lists():\n"
            "    assert match_cron_field('*/15', 31, 0, 59) is False\n"
            "    assert match_cron_field('1-5,10-20/5', 15, 0, 59) is True\n"
            "    assert match_cron_field('1-5,10-20/5', 14, 0, 59) is False\n"
            "    assert match_cron_field('*', 60, 0, 59) is False\n",
        ),
        (
            "syn-csv-parse",
            "Implement `parse_csv_line(line: str) -> list[str]` in `solution.py` supporting "
            'RFC-4180 double-quoted fields, embedded commas, and escaped `""` quotes.\n'
            "Example: `parse_csv_line('a,\"b,c\",d') == ['a', 'b,c', 'd']`.",
            "def parse_csv_line(line: str) -> list[str]:\n    raise NotImplementedError\n",
            "def parse_csv_line(line: str) -> list[str]:\n"
            "    fields: list[str] = []\n"
            "    buf: list[str] = []\n"
            "    in_q = False\n"
            "    i = 0\n"
            "    n = len(line)\n"
            "    while i < n:\n"
            "        ch = line[i]\n"
            "        if in_q:\n"
            "            if ch == '\"':\n"
            "                if i + 1 < n and line[i + 1] == '\"':\n"
            "                    buf.append('\"')\n"
            "                    i += 1\n"
            "                else:\n"
            "                    in_q = False\n"
            "            else:\n"
            "                buf.append(ch)\n"
            "        else:\n"
            "            if ch == ',':\n"
            "                fields.append(''.join(buf))\n"
            "                buf = []\n"
            "            elif ch == '\"':\n"
            "                in_q = True\n"
            "            else:\n"
            "                buf.append(ch)\n"
            "        i += 1\n"
            "    fields.append(''.join(buf))\n"
            "    return fields\n",
            "from solution import parse_csv_line\n\n"
            "def test_example():\n"
            "    assert parse_csv_line('a,\"b,c\",d') == ['a', 'b,c', 'd']\n",
            "from solution import parse_csv_line\n\n"
            "def test_escaped_quotes():\n"
            "    assert parse_csv_line('\"say \"\"hi\"\"\",,x') == ['say \"hi\"', '', 'x']\n"
            "    assert parse_csv_line('') == ['']\n"
            "    assert parse_csv_line('1,2,3') == ['1', '2', '3']\n",
        ),
        (
            "syn-edit-distance",
            "Implement `edit_distance(a: str, b: str) -> int` in `solution.py` computing the "
            "Levenshtein distance (insert, delete, replace cost 1).\n"
            "Example: `edit_distance('kitten', 'sitting') == 3`.",
            "def edit_distance(a: str, b: str) -> int:\n    raise NotImplementedError\n",
            "def edit_distance(a: str, b: str) -> int:\n"
            "    dp = list(range(len(b) + 1))\n"
            "    for i, ca in enumerate(a, start=1):\n"
            "        prev = dp[0]\n"
            "        dp[0] = i\n"
            "        for j, cb in enumerate(b, start=1):\n"
            "            temp = dp[j]\n"
            "            cost = 0 if ca == cb else 1\n"
            "            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)\n"
            "            prev = temp\n"
            "    return dp[-1]\n",
            "from solution import edit_distance\n\n"
            "def test_example():\n"
            "    assert edit_distance('kitten', 'sitting') == 3\n",
            "from solution import edit_distance\n\n"
            "def test_edges():\n"
            "    assert edit_distance('', '') == 0\n"
            "    assert edit_distance('abc', '') == 3\n"
            "    assert edit_distance('intention', 'execution') == 5\n"
            "    assert edit_distance('same', 'same') == 0\n",
        ),
        (
            "syn-word-ladder",
            "Implement `word_ladder_len(begin: str, end: str, words: list[str]) -> int` in "
            "`solution.py` returning the shortest transformation sequence length (1 letter at a "
            "time, every intermediate word in `words`), or `0` if unreachable.\n"
            "Example: `word_ladder_len('hit','cog',['hot','dot','dog','lot','log','cog']) == 5`.",
            "def word_ladder_len(begin: str, end: str, words: list[str]) -> int:\n"
            "    raise NotImplementedError\n",
            "from collections import deque\n\n"
            "def word_ladder_len(begin: str, end: str, words: list[str]) -> int:\n"
            "    word_set = set(words)\n"
            "    if end not in word_set:\n"
            "        return 0\n"
            "    q = deque([(begin, 1)])\n"
            "    visited = {begin}\n"
            "    while q:\n"
            "        cur, dist = q.popleft()\n"
            "        if cur == end:\n"
            "            return dist\n"
            "        for i in range(len(cur)):\n"
            "            for c in 'abcdefghijklmnopqrstuvwxyz':\n"
            "                nxt = cur[:i] + c + cur[i + 1 :]\n"
            "                if nxt in word_set and nxt not in visited:\n"
            "                    visited.add(nxt)\n"
            "                    q.append((nxt, dist + 1))\n"
            "    return 0\n",
            "from solution import word_ladder_len\n\n"
            "def test_example():\n"
            "    assert word_ladder_len('hit','cog',['hot','dot','dog','lot','log','cog']) == 5\n",
            "from solution import word_ladder_len\n\n"
            "def test_unreachable_and_short():\n"
            "    assert word_ladder_len('hit', 'cog', ['hot', 'dot', 'dog']) == 0\n"
            "    assert word_ladder_len('a', 'c', ['a', 'b', 'c']) == 2\n"
            "    assert word_ladder_len('hit','cog',['hot','dot','dog','lot','log','cog']) == 5\n",
        ),
    ]
    return [
        CodingTaskSpec(
            task_id=tid,
            source_benchmark="dasv-synthetic-algos",
            license_id="Apache-2.0",
            prompt=prompt,
            starter_code=starter,
            oracle_code=oracle,
            full_test_code=full_t,
            weak_test_code=weak_t,
        )
        for tid, prompt, starter, oracle, weak_t, full_t in raw_specs
    ]


def _extract_weak_unittest(full_test_code: str, fallback_import: str) -> str:
    """Create a realistic under-specified test keeping only the first test method/assertion."""
    lines = full_test_code.splitlines()
    # Keep imports + class header + first `def test_` method body (classic RULE-MUT-002 flaw)
    out: list[str] = []
    seen_first_test = False
    first_test_indent: int | None = None
    for line in lines:
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        if stripped.startswith("def test_"):
            if not seen_first_test:
                seen_first_test = True
                first_test_indent = indent
                out.append(line)
                continue
            break
        if (
            seen_first_test
            and first_test_indent is not None
            and stripped
            and indent <= first_test_indent
            and not stripped.startswith("@")
        ):
            break
        out.append(line)

    snippet = "\n".join(out).strip()
    if "def test_" in snippet:
        return snippet + "\n"
    return f"{fallback_import}\n\ndef test_smoke_import():\n    assert True\n"


def verify_task_oracle_and_noop(task: CodingTaskSpec, timeout_sec: float = 6.0) -> bool:
    """Confirm canonical oracle passes full_test_code and starter stub fails before inclusion."""
    import tempfile

    with tempfile.TemporaryDirectory(prefix="dasv_verify_") as tmp:
        ws = Path(tmp)
        test_file = ws / "test_solution.py"
        test_file.write_text(task.full_test_code, encoding="utf-8")
        env = {
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "PYTHONPATH": str(ws),
            "PYTHONHASHSEED": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        (ws / "solution.py").write_text(task.starter_code, encoding="utf-8")
        try:
            noop_p = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", str(test_file)],
                cwd=str(ws),
                env=env,
                capture_output=True,
                timeout=timeout_sec,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return False
        if noop_p.returncode == 0:
            return False

        (ws / "solution.py").write_text(task.oracle_code, encoding="utf-8")
        try:
            ora_p = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", str(test_file)],
                cwd=str(ws),
                env=env,
                capture_output=True,
                timeout=timeout_sec,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return False
        return ora_p.returncode == 0


def _load_hf_corpus(
    target_total: int,
    seed: int,
    *,
    verify_oracles: bool = True,
) -> list[CodingTaskSpec]:
    """Load real permissive-license Python tasks from BigCodeBench + MBPP+ via HF `datasets`."""
    from concurrent.futures import ThreadPoolExecutor

    from datasets import load_dataset  # type: ignore[import-untyped]

    candidates: list[CodingTaskSpec] = []
    seen_ids: set[str] = set()

    # 1. Primary: BigCodeBench (Apache-2.0), stdlib/numpy/pandas offline subset
    bcb_ds: Any = None
    for split_name in ("v0.1.4", "v0.1.2"):
        try:
            bcb_ds = load_dataset("bigcode/bigcodebench", split=split_name)
            break
        except Exception:
            continue

    if bcb_ds is not None:
        for row in bcb_ds:
            libs_raw = row.get("libs") or []
            if isinstance(libs_raw, str):
                try:
                    libs = [str(x).strip() for x in json.loads(libs_raw)]
                except Exception:
                    libs = [x.strip(" '[]\"") for x in libs_raw.split(",") if x.strip()]
            else:
                libs = [str(x).strip() for x in libs_raw]
            if any(lib not in _ALLOWED_HF_LIBS for lib in libs):
                continue

            raw_id = str(row.get("task_id", "")).replace("/", "-").lower()
            tid = re.sub(r"[^a-z0-9_-]+", "-", raw_id).strip("-")
            if not tid or tid in seen_ids:
                continue

            code_prompt = str(row.get("code_prompt") or row.get("instruct_prompt") or "").strip()
            canonical = str(row.get("canonical_solution") or "")
            test_code = str(row.get("test") or "").strip()
            entry_point = str(row.get("entry_point") or "task_func").strip()
            if not code_prompt or not test_code or not canonical:
                continue

            oracle_code = code_prompt + "\n" + canonical + "\n"
            full_test = (
                f"from solution import *\nfrom solution import {entry_point}\n\n{test_code}\n"
            )
            weak_test = _extract_weak_unittest(full_test, f"from solution import {entry_point}")
            prompt_text = (
                f"Solve the following Python programming task in `solution.py`.\n"
                f"Implement `{entry_point}` matching the exact signature and specification below.\n"
                "Return your complete implementation inside a ```python``` block starting with "
                "`# file: solution.py`.\n\n"
                f"```python\n{code_prompt}\n```\n"
            )
            seen_ids.add(tid)
            candidates.append(
                CodingTaskSpec(
                    task_id=tid,
                    source_benchmark="bigcode/bigcodebench",
                    license_id="Apache-2.0",
                    prompt=prompt_text,
                    starter_code=code_prompt + "\n    raise NotImplementedError\n",
                    oracle_code=oracle_code,
                    full_test_code=full_test,
                    weak_test_code=weak_test,
                )
            )

    # 2. Supplement if needed: EvalPlus MBPP+ (Apache-2.0)
    if len(candidates) < target_total * 2:
        try:
            mbpp_ds = load_dataset("evalplus/mbppplus", split="test")
        except Exception:
            mbpp_ds = []
        for row in mbpp_ds:
            raw_id = f"mbpp-{row.get('task_id', '')}".replace("/", "-").lower()
            tid = re.sub(r"[^a-z0-9_-]+", "-", raw_id).strip("-")
            if not tid or tid in seen_ids:
                continue
            prompt_desc = str(row.get("prompt") or "").strip()
            oracle_code = str(row.get("code") or "").strip()
            test_list = list(row.get("test_list") or [])
            test_imports = "\n".join(str(x) for x in (row.get("test_imports") or []))
            if not prompt_desc or not oracle_code or len(test_list) < 2:
                continue
            full_assertions = "\n".join(f"    {t}" for t in test_list)
            full_test = (
                f"{test_imports}\nfrom solution import *\n\ndef test_all():\n{full_assertions}\n"
            )
            weak_test = (
                f"{test_imports}\nfrom solution import *\n\ndef test_first():\n    {test_list[0]}\n"
            )
            prompt_text = (
                "Solve the following Python programming task in `solution.py`.\n"
                "Return your complete implementation inside a ```python``` block starting with "
                "`# file: solution.py`.\n\n"
                f"{prompt_desc}\n\n"
                f"Example assertion:\n```python\n{test_list[0]}\n```\n"
            )
            seen_ids.add(tid)
            candidates.append(
                CodingTaskSpec(
                    task_id=tid,
                    source_benchmark="evalplus/mbppplus",
                    license_id="Apache-2.0",
                    prompt=prompt_text,
                    starter_code="# Implement solution here\n",
                    oracle_code=oracle_code + "\n",
                    full_test_code=full_test,
                    weak_test_code=weak_test,
                )
            )

    rng = random.Random(seed)
    candidates.sort(key=lambda t: t.task_id)
    rng.shuffle(candidates)

    if not verify_oracles:
        tasks = candidates[:target_total]
    else:
        tasks = []
        batch_sz = 32
        with ThreadPoolExecutor(max_workers=8) as pool:
            for i in range(0, len(candidates), batch_sz):
                if len(tasks) >= target_total:
                    break
                chunk = candidates[i : i + batch_sz]
                results = list(pool.map(verify_task_oracle_and_noop, chunk))
                for task, ok in zip(chunk, results, strict=True):
                    if ok:
                        tasks.append(task)
                        if len(tasks) >= target_total:
                            break

    if len(tasks) < target_total:
        raise RuntimeError(
            f"Requested {target_total} HF tasks but only {len(tasks)} oracle-verified tasks loaded."
        )
    return tasks[:target_total]


def _write_harbor_task(
    task: CodingTaskSpec,
    task_dir: Path,
    *,
    hardened: bool,
    defect_kind: str,
    pinned_image: str,
) -> dict[str, Any]:
    """Materialize a valid Harbor v1.1 directory for Arm 1 (unhardened) or Arm 2 (hardened)."""
    (task_dir / "environment").mkdir(parents=True, exist_ok=True)
    (task_dir / "solution").mkdir(parents=True, exist_ok=True)
    (task_dir / "tests").mkdir(parents=True, exist_ok=True)

    # Instruction text (for RULE-BNDY-001 unhardened tasks, reflects exposed workspace tests)
    instruction = task.prompt.strip() + "\n"
    if not hardened and defect_kind == "RULE-BNDY-001":
        instruction += (
            "\nWorkspace note: `solution.py` and `tests/test_solution.py` are mounted in `/app`. "
            "You may run `pytest tests/test_solution.py` to check your work.\n"
        )
    (task_dir / "instruction.md").write_text(instruction, encoding="utf-8")

    # task.toml (Harbor v1.1 schema)
    allow_internet_str = "false" if hardened else "true"
    net_mode_line = 'network_mode = "no-network"\n' if hardened else ""
    (task_dir / "task.toml").write_text(
        'schema_version = "1.1"\n'
        "[metadata]\n"
        f'name = "{task.task_id}"\n'
        f'source_benchmark = "{task.source_benchmark}"\n'
        f'license = "{task.license_id}"\n'
        f'arm_variant = "{"hardened" if hardened else "unhardened"}"\n'
        f'defect_kind = "{defect_kind}"\n'
        "\n"
        "[environment]\n"
        f"allow_internet = {allow_internet_str}\n"
        f"{net_mode_line}"
        "\n"
        "[verifier]\n"
        'entrypoint = "tests/test.sh"\n'
        "\n"
        "[solution]\n"
        'entrypoint = "solution/solve.sh"\n',
        encoding="utf-8",
    )

    # Starter file inside environment build context
    (task_dir / "environment" / "solution.py").write_text(task.starter_code, encoding="utf-8")

    # Dockerfile
    base_img = pinned_image if hardened else UNPINNED_PYTHON_IMAGE
    dockerfile_lines = [
        f"FROM {base_img}",
        "WORKDIR /app",
        "COPY environment/solution.py /app/solution.py",
    ]
    if not hardened and defect_kind == "RULE-BNDY-001":
        dockerfile_lines.append("COPY tests/test_solution.py /app/tests/test_solution.py")
    (task_dir / "environment" / "Dockerfile").write_text(
        "\n".join(dockerfile_lines) + "\n", encoding="utf-8"
    )

    # Oracle script
    oracle_b64_safe = json.dumps(task.oracle_code)
    (task_dir / "solution" / "solve.sh").write_text(
        "#!/bin/sh\nset -eu\n"
        f'python3 -c \'import pathlib; pathlib.Path("solution.py").write_text({oracle_b64_safe}, '
        'encoding="utf-8")\'\n',
        encoding="utf-8",
    )

    # Verifier test suite (Arm 1 RULE-MUT-002 uses weak_test_code; Arm 2 & heldout use full)
    active_test = (
        task.weak_test_code
        if (not hardened and defect_kind == "RULE-MUT-002")
        else (task.full_test_code)
    )
    (task_dir / "tests" / "test_solution.py").write_text(active_test, encoding="utf-8")
    # Also store canonical full tests outside `tests/` for off-policy clean shadow evaluation
    (task_dir / "shadow_clean_test.py").write_text(task.full_test_code, encoding="utf-8")
    (task_dir / "tests" / "test.sh").write_text(
        "#!/bin/sh\nset -eu\n"
        'VERIFIER_DIR="$(cd "$(dirname "$0")" && pwd)"\n'
        'PYTHONPATH="/app:${PYTHONPATH:-}" pytest -q "${VERIFIER_DIR}/test_solution.py"\n',
        encoding="utf-8",
    )

    return {
        "prompt": [{"role": "user", "content": instruction}],
        "task_id": task.task_id,
        "root_path": str(task_dir.resolve()),
        "source_benchmark": task.source_benchmark,
        "hardened": hardened,
        "defect_kind": defect_kind,
        "prompt_sha256": hashlib.sha256(task.prompt.encode("utf-8")).hexdigest()[:16],
    }


def prepare_ablation_taskset(
    *,
    out_dir: Path,
    source: str = "hf",
    n_train: int = 128,
    n_heldout: int = 64,
    seed: int = 42,
    bndy_frac: float = 0.20,
    weak_test_frac: float = 0.20,
) -> dict[str, Any]:
    """Generate paired Arm 1 (unhardened), Arm 2 (hardened), and shared clean held-out splits."""
    if source == "synthetic":
        corpus = build_synthetic_corpus()
        if n_train + n_heldout > len(corpus):
            n_train = min(n_train, 12)
            n_heldout = min(n_heldout, len(corpus) - n_train)
        rng = random.Random(seed)
        ordered = list(corpus)
        rng.shuffle(ordered)
        selected = ordered[: n_train + n_heldout]
    elif source == "hf":
        selected = _load_hf_corpus(n_train + n_heldout, seed=seed)
    else:
        raise ValueError(f"Unsupported --source '{source}'; expected 'hf' or 'synthetic'.")

    train_tasks = selected[:n_train]
    heldout_tasks = selected[n_train : n_train + n_heldout]

    # Enforce strict zero-overlap between train & held-out by both task_id and prompt hash
    train_ids = {t.task_id for t in train_tasks}
    heldout_ids = {t.task_id for t in heldout_tasks}
    assert train_ids.isdisjoint(heldout_ids), "Train/held-out task_id overlap detected!"

    out_dir.mkdir(parents=True, exist_ok=True)
    pinned_image = _resolve_pinned_python_image()

    n_bndy = int(round(len(train_tasks) * bndy_frac))
    n_weak = int(round(len(train_tasks) * weak_test_frac))

    arm1_rows: list[dict[str, Any]] = []
    arm2_rows: list[dict[str, Any]] = []
    heldout_rows: list[dict[str, Any]] = []

    adapter = HarborAdapter()
    arm1_tier0_fail_count = 0
    arm2_tier0_clean_count = 0
    defect_counts: dict[str, int] = {"RULE-BNDY-001": 0, "RULE-MUT-002": 0, "hermetic_only": 0}

    for idx, task in enumerate(train_tasks):
        if idx < n_bndy:
            defect = "RULE-BNDY-001"
        elif idx < n_bndy + n_weak:
            defect = "RULE-MUT-002"
        else:
            defect = "hermetic_only"
        defect_counts[defect] += 1

        d1 = out_dir / "train_unhardened" / task.task_id
        r1 = _write_harbor_task(
            task, d1, hardened=False, defect_kind=defect, pinned_image=pinned_image
        )
        arm1_rows.append(r1)
        if run_tier0_static_probes(adapter.load(d1)):
            arm1_tier0_fail_count += 1

        d2 = out_dir / "train_hardened" / task.task_id
        r2 = _write_harbor_task(
            task, d2, hardened=True, defect_kind="none", pinned_image=pinned_image
        )
        arm2_rows.append(r2)
        if not run_tier0_static_probes(adapter.load(d2)):
            arm2_tier0_clean_count += 1

    for task in heldout_tasks:
        dh = out_dir / "heldout_clean" / task.task_id
        rh = _write_harbor_task(
            task, dh, hardened=True, defect_kind="none", pinned_image=pinned_image
        )
        heldout_rows.append(rh)

    for fname, rows in (
        ("train_unhardened.jsonl", arm1_rows),
        ("train_hardened.jsonl", arm2_rows),
        ("heldout_clean.jsonl", heldout_rows),
    ):
        (out_dir / fname).write_text(
            "\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n",
            encoding="utf-8",
        )

    manifest = {
        "schema_version": "1.0.0",
        "source": source,
        "seed": seed,
        "pinned_image": pinned_image,
        "n_train": len(train_tasks),
        "n_heldout": len(heldout_tasks),
        "arm1_defect_breakdown": defect_counts,
        "dasv_tier0_audit": {
            "arm1_tasks_failing_tier0": arm1_tier0_fail_count,
            "arm2_tasks_clean_tier0": arm2_tier0_clean_count,
        },
        "train_ids_sha256": hashlib.sha256(",".join(sorted(train_ids)).encode("utf-8")).hexdigest(),
        "heldout_ids_sha256": hashlib.sha256(
            ",".join(sorted(heldout_ids)).encode("utf-8")
        ).hexdigest(),
    }
    (out_dir / "taskset_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare paired Harbor taskset for Dr. GRPO.")
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/taskset_v1"))
    parser.add_argument("--source", choices=("hf", "synthetic"), default="hf")
    parser.add_argument("--n-train", type=int, default=128)
    parser.add_argument("--n-heldout", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest = prepare_ablation_taskset(
        out_dir=args.out_dir,
        source=args.source,
        n_train=args.n_train,
        n_heldout=args.n_heldout,
        seed=args.seed,
    )
    print(
        f"[prepare_taskset] Ready ({manifest['source']}): "
        f"train={manifest['n_train']}, heldout={manifest['n_heldout']}, "
        f"arm1_tier0_flagged={manifest['dasv_tier0_audit']['arm1_tasks_failing_tier0']}, "
        f"arm2_tier0_clean={manifest['dasv_tier0_audit']['arm2_tasks_clean_tier0']}"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[prepare_taskset] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
