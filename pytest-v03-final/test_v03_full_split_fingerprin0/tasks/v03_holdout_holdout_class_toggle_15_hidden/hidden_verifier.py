import os
import sys
sys.path.insert(0, os.environ.get('JARVIS_WORKSPACE', '/workspace'))
from solution import Toggle
t = Toggle(); t.flip(); t.flip(); assert t.state() is False
