import unittest
from solution import transpose

class PublicTests(unittest.TestCase):
    def test_public(self):
        self.assertEqual(transpose([[1, 2], [3, 4]]), [[1, 3], [2, 4]])

if __name__ == '__main__':
    unittest.main()
