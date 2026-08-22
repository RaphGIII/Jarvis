import unittest
from solution import is_anagram

class PublicTests(unittest.TestCase):
    def test_public(self):
        self.assertTrue(is_anagram('listen', 'silent'))

if __name__ == '__main__':
    unittest.main()
