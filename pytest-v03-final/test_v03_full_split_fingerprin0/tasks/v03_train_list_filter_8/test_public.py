import unittest
from solution import positive_values

class PublicTests(unittest.TestCase):
    def test_public(self):
        self.assertEqual(positive_values([-1, 2, 3]), [2, 3])

if __name__ == '__main__':
    unittest.main()
