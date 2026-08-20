import os
import sys
sys.path.insert(0, os.environ.get('JARVIS_WORKSPACE', '/workspace'))
from solution import Stack
s = Stack(); s.push('a'); s.push('b'); assert s.pop() == 'b'
