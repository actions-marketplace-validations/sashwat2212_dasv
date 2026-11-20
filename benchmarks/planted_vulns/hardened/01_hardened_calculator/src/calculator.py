def add(a: int, b: int) -> int:
    if a < 0 or b < 0:
        raise ValueError("inputs must be non-negative")
    # Buggy baseline: subtracts instead of adding (fixed by solution/solve.sh)
    return a - b

