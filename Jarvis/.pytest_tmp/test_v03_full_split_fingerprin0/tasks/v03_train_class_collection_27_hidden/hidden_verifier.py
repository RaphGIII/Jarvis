import os
import sys
sys.path.insert(0, os.environ.get('JARVIS_WORKSPACE', '/workspace'))
from solution import Bag
b = Bag(); b.add('a'); b.add('b'); assert b.size() == 2
