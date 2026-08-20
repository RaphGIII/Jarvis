import os
import sys
sys.path.insert(0, os.environ.get('JARVIS_WORKSPACE', '/workspace'))
from solution import fib
assert fib(0) == 0 and fib(1) == 1
