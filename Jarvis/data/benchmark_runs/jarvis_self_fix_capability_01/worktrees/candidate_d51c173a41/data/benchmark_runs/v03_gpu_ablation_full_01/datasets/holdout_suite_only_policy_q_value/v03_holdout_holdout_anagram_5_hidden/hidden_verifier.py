import os
import sys
sys.path.insert(0, os.environ.get('JARVIS_WORKSPACE', '/workspace'))
from solution import is_anagram
assert not is_anagram('abc', 'abd')
