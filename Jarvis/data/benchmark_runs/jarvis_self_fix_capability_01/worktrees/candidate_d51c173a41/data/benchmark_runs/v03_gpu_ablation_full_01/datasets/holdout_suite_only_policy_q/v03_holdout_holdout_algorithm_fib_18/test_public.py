import unittest
from solution import fib

class PublicTests(unittest.TestCase):
    def test_public(self):
        self.assertEqual(fib(4), 3)

if __name__ == '__main__':
    unittest.main()
