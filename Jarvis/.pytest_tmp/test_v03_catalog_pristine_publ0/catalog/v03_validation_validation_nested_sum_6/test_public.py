import unittest
from solution import sum_matrix

class PublicTests(unittest.TestCase):
    def test_public(self):
        self.assertEqual(sum_matrix([[1, 2], [3]]), 6)

if __name__ == '__main__':
    unittest.main()
