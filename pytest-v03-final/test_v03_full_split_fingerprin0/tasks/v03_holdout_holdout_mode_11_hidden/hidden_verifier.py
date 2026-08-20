import os
import sys
sys.path.insert(0, os.environ.get('JARVIS_WORKSPACE', '/workspace'))
from solution import most_common
assert most_common([2, 2, 3]) == 2
