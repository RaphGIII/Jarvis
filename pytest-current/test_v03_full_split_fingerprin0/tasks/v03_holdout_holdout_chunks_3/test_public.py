import unittest
from solution import chunk_pairs

class PublicTests(unittest.TestCase):
    def test_public(self):
        self.assertEqual(chunk_pairs([1, 2, 3, 4]), [[1, 2], [3, 4]])

if __name__ == '__main__':
    unittest.main()
