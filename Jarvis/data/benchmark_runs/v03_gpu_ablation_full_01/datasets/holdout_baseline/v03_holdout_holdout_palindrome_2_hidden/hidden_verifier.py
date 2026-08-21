import os
import sys
sys.path.insert(0, os.environ.get('JARVIS_WORKSPACE', '/workspace'))
from solution import is_palindrome
assert not is_palindrome('jarvis')
