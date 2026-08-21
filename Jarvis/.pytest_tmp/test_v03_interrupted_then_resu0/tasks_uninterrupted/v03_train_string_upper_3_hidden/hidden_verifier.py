import os
import sys
sys.path.insert(0, os.environ.get('JARVIS_WORKSPACE', '/workspace'))
from solution import clean_upper
assert clean_upper('\tbob ') == 'BOB'
