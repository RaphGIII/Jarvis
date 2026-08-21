import os
import sys
sys.path.insert(0, os.environ.get('JARVIS_WORKSPACE', '/workspace'))
from solution import parse_pairs
assert parse_pairs('x=9') == {'x': '9'}
