import os
import sys
sys.path.insert(0, os.environ.get('JARVIS_WORKSPACE', '/workspace'))
from solution import first_index
assert first_index([4, 5], 5) == 1
