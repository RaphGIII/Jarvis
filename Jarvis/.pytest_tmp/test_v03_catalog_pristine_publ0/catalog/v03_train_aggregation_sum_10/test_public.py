import unittest
from solution import sum_even

class PublicTests(unittest.TestCase):
    def test_public(self):
        self.assertEqual(sum_even([1, 2, 4]), 6)

if __name__ == '__main__':
    unittest.main()
