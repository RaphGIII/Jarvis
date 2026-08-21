import unittest
from solution import add_numbers

class PublicTests(unittest.TestCase):
    def test_public(self):
        self.assertEqual(add_numbers(2, 3), 5)

if __name__ == '__main__':
    unittest.main()
