import unittest
from solution import most_common

class PublicTests(unittest.TestCase):
    def test_public(self):
        self.assertEqual(most_common(['a', 'b', 'a']), 'a')

if __name__ == '__main__':
    unittest.main()
