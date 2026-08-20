import os
import sys
sys.path.insert(0, os.environ.get('JARVIS_WORKSPACE', '/workspace'))
from solution import dedupe_keep_order
assert dedupe_keep_order([]) == []
