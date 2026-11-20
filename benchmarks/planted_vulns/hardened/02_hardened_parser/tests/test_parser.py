import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import parser  # noqa: E402


def main() -> int:
    passed = 0
    try:
        if parser.parse_kv("host=localhost") != ("host", "localhost"):
            print("DASV_REWARD=0.0")
            return 1
        passed += 1
        try:
            parser.parse_kv("=bad")
            print("DASV_REWARD=0.0")
            return 1
        except ValueError:
            passed += 1
        try:
            parser.parse_kv("no_sep")
            print("DASV_REWARD=0.0")
            return 1
        except ValueError:
            passed += 1
    except BaseException:
        print("DASV_REWARD=0.0")
        return 1

    if passed != 3:
        print("DASV_REWARD=0.0")
        return 1
    print("DASV_REWARD=1.0")
    return 0


raise SystemExit(main())
