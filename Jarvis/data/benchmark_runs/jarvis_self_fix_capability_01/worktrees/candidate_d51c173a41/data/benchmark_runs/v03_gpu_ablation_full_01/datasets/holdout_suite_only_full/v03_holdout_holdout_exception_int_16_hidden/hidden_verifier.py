import os
import sys
sys.path.insert(0, os.environ.get('JARVIS_WORKSPACE', '/workspace'))
from solution import parse_optional_int
assert parse_optional_int('7') == 7
