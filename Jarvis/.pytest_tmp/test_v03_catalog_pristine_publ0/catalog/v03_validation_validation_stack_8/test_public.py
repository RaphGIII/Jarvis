import unittest
from solution import Stack

class PublicTests(unittest.TestCase):
    def test_public(self):
        s = Stack(); s.push(3); self.assertEqual(s.pop(), 3)

if __name__ == '__main__':
    unittest.main()
