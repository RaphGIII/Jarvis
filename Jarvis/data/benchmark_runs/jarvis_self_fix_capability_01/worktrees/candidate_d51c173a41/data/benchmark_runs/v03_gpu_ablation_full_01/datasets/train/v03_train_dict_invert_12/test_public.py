import unittest
from solution import invert_lookup

class PublicTests(unittest.TestCase):
    def test_public(self):
        self.assertEqual(invert_lookup({'a': 1}), {1: 'a'})

if __name__ == '__main__':
    unittest.main()
