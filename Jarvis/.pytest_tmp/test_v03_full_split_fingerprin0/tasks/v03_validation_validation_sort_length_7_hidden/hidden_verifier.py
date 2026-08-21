import os
import sys
sys.path.insert(0, os.environ.get('JARVIS_WORKSPACE', '/workspace'))
from solution import sort_by_length
assert sort_by_length(['ba', 'ab']) == ['ab', 'ba']
