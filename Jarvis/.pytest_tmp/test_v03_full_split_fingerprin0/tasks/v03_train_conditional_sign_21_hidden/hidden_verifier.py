import os
import sys
sys.path.insert(0, os.environ.get('JARVIS_WORKSPACE', '/workspace'))
from solution import sign_label
assert sign_label(-1) == 'negative'
