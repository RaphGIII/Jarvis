import unittest
from solution import Counter

class PublicTests(unittest.TestCase):
    def test_public(self):
        c = Counter(); c.increment(); self.assertEqual(c.value(), 1)

if __name__ == '__main__':
    unittest.main()
