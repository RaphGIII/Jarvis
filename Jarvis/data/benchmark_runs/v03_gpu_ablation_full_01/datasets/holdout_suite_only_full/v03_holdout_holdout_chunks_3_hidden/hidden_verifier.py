import os
import sys
sys.path.insert(0, os.environ.get('JARVIS_WORKSPACE', '/workspace'))
from solution import chunk_pairs
assert chunk_pairs([1]) == [[1]]
