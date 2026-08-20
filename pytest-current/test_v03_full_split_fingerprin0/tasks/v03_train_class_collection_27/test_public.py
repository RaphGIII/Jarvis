import unittest
from solution import Bag

class PublicTests(unittest.TestCase):
    def test_public(self):
        b = Bag(); b.add('x'); self.assertEqual(b.size(), 1)

if __name__ == '__main__':
    unittest.main()
