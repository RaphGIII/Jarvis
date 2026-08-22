import os
import sys
sys.path.insert(0, os.environ.get('JARVIS_WORKSPACE', '/workspace'))
from solution import title_words
assert title_words(' ada ') == 'Ada'
