import os
import sys
sys.path.insert(0, os.environ.get('JARVIS_WORKSPACE', '/workspace'))
from solution import min_max
assert min_max([-2]) == (-2, -2)
