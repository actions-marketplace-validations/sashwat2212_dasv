def solve(x: int) -> int:
    if x < 0:
        raise ValueError('x must be >= 0')
    return x * x
