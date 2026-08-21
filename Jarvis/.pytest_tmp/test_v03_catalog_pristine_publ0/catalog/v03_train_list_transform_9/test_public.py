import unittest
from solution import double_values

class PublicTests(unittest.TestCase):
    def test_public(self):
        self.assertEqual(double_values([1, 3]), [2, 6])

if __name__ == '__main__':
    unittest.main()
