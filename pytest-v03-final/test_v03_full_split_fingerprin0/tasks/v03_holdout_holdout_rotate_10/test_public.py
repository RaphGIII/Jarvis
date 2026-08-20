import unittest
from solution import rotate_left

class PublicTests(unittest.TestCase):
    def test_public(self):
        self.assertEqual(rotate_left([1, 2, 3], 1), [2, 3, 1])

if __name__ == '__main__':
    unittest.main()
