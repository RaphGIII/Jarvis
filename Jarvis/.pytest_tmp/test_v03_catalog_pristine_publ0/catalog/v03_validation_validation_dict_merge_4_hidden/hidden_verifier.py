import os
import sys
sys.path.insert(0, os.environ.get('JARVIS_WORKSPACE', '/workspace'))
from solution import merge_counts
assert merge_counts({}, {'x': 4}) == {'x': 4}
