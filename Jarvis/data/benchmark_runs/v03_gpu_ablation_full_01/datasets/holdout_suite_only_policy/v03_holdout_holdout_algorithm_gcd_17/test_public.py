import unittest
from solution import gcd

class PublicTests(unittest.TestCase):
    def test_public(self):
        self.assertEqual(gcd(12, 8), 4)

if __name__ == '__main__':
    unittest.main()
