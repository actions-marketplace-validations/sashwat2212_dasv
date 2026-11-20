def add(a: int, b: int) -> int:
    if a < 0:
        raise ValueError('negative input')
    return a + b
