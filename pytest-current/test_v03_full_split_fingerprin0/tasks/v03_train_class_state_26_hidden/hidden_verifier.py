import os
import sys
sys.path.insert(0, os.environ.get('JARVIS_WORKSPACE', '/workspace'))
from solution import Counter
c = Counter()
for _ in range(3): c.increment()
assert c.value() == 3
