import os
import sys
sys.path.insert(0, os.environ.get('JARVIS_WORKSPACE', '/workspace'))
from solution import add_numbers
assert add_numbers(-2, 5) == 3
