import os
import sys
sys.path.insert(0, os.environ.get('JARVIS_WORKSPACE', '/workspace'))
from solution import square_plus_one
assert square_plus_one(0) == 1
