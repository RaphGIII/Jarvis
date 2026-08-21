import unittest
from solution import factorial

class PublicTests(unittest.TestCase):
    def test_public(self):
        self.assertEqual(factorial(4), 24)

if __name__ == '__main__':
    unittest.main()
