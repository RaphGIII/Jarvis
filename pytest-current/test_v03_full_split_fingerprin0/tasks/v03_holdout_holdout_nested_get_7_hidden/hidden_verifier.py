import os
import sys
sys.path.insert(0, os.environ.get('JARVIS_WORKSPACE', '/workspace'))
from solution import nested_get
assert nested_get({'a': {'b': 2}}, 'a', 'b', 0) == 2
