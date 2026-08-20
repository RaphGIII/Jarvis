import unittest
from solution import flatten_once

class PublicTests(unittest.TestCase):
    def test_public(self):
        self.assertEqual(flatten_once([[1, 2], [3]]), [1, 2, 3])

if __name__ == '__main__':
    unittest.main()
