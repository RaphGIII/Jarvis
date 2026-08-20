import os
import sys
sys.path.insert(0, os.environ.get('JARVIS_WORKSPACE', '/workspace'))
from solution import make_initials
assert make_initials('Grace Brewster Hopper') == 'GBH'
