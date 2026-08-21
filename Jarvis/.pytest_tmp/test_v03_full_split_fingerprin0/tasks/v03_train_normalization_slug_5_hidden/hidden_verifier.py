import os
import sys
sys.path.insert(0, os.environ.get('JARVIS_WORKSPACE', '/workspace'))
from solution import slugify
assert slugify(' Two  Words ') == 'two-words'
