import os
import sys
sys.path.insert(0, os.environ.get('JARVIS_WORKSPACE', '/workspace'))
from solution import clamp
assert clamp(11, 0, 10) == 10
