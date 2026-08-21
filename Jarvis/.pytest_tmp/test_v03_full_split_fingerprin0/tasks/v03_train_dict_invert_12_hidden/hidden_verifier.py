import os
import sys
sys.path.insert(0, os.environ.get('JARVIS_WORKSPACE', '/workspace'))
from solution import invert_lookup
assert invert_lookup({'x': 'y'}) == {'y': 'x'}
