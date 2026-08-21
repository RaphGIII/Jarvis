import os
import sys
sys.path.insert(0, os.environ.get('JARVIS_WORKSPACE', '/workspace'))
from solution import even_values
assert even_values([0, -2, 5]) == [0, -2]
