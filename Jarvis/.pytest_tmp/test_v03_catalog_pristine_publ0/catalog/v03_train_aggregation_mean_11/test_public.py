import unittest
from solution import mean_or_zero

class PublicTests(unittest.TestCase):
    def test_public(self):
        self.assertEqual(mean_or_zero([]), 0)

if __name__ == '__main__':
    unittest.main()
