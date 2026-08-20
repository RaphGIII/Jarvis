import os
import sys
sys.path.insert(0, os.environ.get('JARVIS_WORKSPACE', '/workspace'))
from solution import sort_desc
assert sort_desc([]) == []
