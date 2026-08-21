import unittest
from solution import is_palindrome

class PublicTests(unittest.TestCase):
    def test_public(self):
        self.assertTrue(is_palindrome('Never odd or even'))

if __name__ == '__main__':
    unittest.main()
