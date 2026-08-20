import unittest
from calculator import combine_values

class PublicTests(unittest.TestCase):
    def test_public_addition(self):
        self.assertEqual(combine_values(2, 3), 5)

if __name__ == '__main__':
    unittest.main()
