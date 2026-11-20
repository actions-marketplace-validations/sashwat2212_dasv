import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
import solver
# Weak in-process runner: relies solely on exit code 0 without counting completed assertions!
assert solver.solve(4) == 16
assert solver.solve(5) == 25
