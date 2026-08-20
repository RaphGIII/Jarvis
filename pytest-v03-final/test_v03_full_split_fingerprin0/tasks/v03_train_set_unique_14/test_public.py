import unittest
from solution import unique_sorted

class PublicTests(unittest.TestCase):
    def test_public(self):
        self.assertEqual(unique_sorted([3, 1, 3]), [1, 3])

if __name__ == '__main__':
    unittest.main()
