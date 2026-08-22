import unittest
from solution import merge_sorted

class PublicTests(unittest.TestCase):
    def test_public(self):
        self.assertEqual(merge_sorted([1, 3], [2]), [1, 2, 3])

if __name__ == '__main__':
    unittest.main()
