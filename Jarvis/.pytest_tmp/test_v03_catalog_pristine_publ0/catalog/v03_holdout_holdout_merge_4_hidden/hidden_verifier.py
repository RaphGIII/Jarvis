import os
import sys
sys.path.insert(0, os.environ.get('JARVIS_WORKSPACE', '/workspace'))
from solution import merge_sorted
assert merge_sorted([], [1]) == [1]
