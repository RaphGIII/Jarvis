import os
import sys
sys.path.insert(0, os.environ.get('JARVIS_WORKSPACE', '/workspace'))
from solution import safe_divide
assert safe_divide(6, 3) == 2
