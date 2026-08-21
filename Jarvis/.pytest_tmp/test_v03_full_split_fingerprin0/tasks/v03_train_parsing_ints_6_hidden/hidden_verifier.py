import os
import sys
sys.path.insert(0, os.environ.get('JARVIS_WORKSPACE', '/workspace'))
from solution import parse_ints
assert parse_ints('10,-2') == [10, -2]
