import os
import sys
sys.path.insert(0, os.environ.get('JARVIS_WORKSPACE', '/workspace'))
from solution import reverse_text
assert reverse_text('Jarvis') == 'sivraJ'
