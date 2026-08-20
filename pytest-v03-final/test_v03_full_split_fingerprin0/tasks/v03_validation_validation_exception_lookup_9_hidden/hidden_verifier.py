import os
import sys
sys.path.insert(0, os.environ.get('JARVIS_WORKSPACE', '/workspace'))
from solution import lookup_default
assert lookup_default({'x': 1}, 'x', 0) == 1
