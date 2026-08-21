import os
import sys
sys.path.insert(0, os.environ.get('JARVIS_WORKSPACE', '/workspace'))
from solution import mean_or_zero
assert mean_or_zero([2, 4]) == 3
