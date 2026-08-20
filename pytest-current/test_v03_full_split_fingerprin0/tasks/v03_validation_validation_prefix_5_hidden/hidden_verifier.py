import os
import sys
sys.path.insert(0, os.environ.get('JARVIS_WORKSPACE', '/workspace'))
from solution import starts_with_any
assert not starts_with_any('alpha', [])
