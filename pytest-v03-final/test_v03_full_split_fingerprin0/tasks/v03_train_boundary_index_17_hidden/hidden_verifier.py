import os
import sys
sys.path.insert(0, os.environ.get('JARVIS_WORKSPACE', '/workspace'))
from solution import safe_get
assert safe_get([7], 0, None) == 7
