import os
import sys
sys.path.insert(0, os.environ.get('JARVIS_WORKSPACE', '/workspace'))
from solution import positive_values
assert positive_values([-5, 0, 4]) == [4]
