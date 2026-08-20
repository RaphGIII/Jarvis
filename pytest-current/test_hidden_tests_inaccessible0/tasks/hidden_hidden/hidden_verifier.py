import os
import sys
sys.path.insert(0, os.environ.get('JARVIS_WORKSPACE', '/workspace'))
from calculator import add
assert add(-2, 5) == 3
assert add(10, 7) == 17
