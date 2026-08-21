import os
import sys
sys.path.insert(0, os.environ.get('JARVIS_WORKSPACE', '/workspace'))
from solution import absolute_delta
assert absolute_delta(9, 4) == 5
