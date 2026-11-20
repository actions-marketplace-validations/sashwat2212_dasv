import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
import calculator
passed = 0
if calculator.add(2, 3) != 5:
    sys.exit(1)
passed += 1
if calculator.add(10, 20) != 30:
    sys.exit(1)
passed += 1
try:
    calculator.add(-1, 5)
    sys.exit(1)
except ValueError:
    passed += 1
assert passed == 3
print('DASV_REWARD=1.0')
