import unittest
from solution import even_values

class PublicTests(unittest.TestCase):
    def test_public(self):
        self.assertEqual(even_values([1, 2, 3, 4]), [2, 4])

if __name__ == '__main__':
    unittest.main()
