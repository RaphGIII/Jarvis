import os
import sys
sys.path.insert(0, os.environ.get('JARVIS_WORKSPACE', '/workspace'))
from solution import sum_matrix
assert sum_matrix([]) == 0
