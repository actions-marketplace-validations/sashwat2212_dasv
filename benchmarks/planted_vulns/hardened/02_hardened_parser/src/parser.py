def parse_kv(line: str) -> tuple[str, str]:
    if "=" not in line:
        raise ValueError("missing =")
    k, _v = line.split("=", 1)
    if not k.strip():
        raise ValueError("empty key")
    # Buggy baseline: drops value (fixed by solution/solve.sh)
    return k.strip(), ""

