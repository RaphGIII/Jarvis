import os
import sys
sys.path.insert(0, os.environ.get('JARVIS_WORKSPACE', '/workspace'))
from calculator import combine_values
assert combine_values(-2, 5) == 3
assert combine_values(10, 7) == 17
