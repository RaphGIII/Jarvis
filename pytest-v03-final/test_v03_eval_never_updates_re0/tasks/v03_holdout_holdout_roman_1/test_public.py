import unittest
from solution import roman_one_to_three

class PublicTests(unittest.TestCase):
    def test_public(self):
        self.assertEqual(roman_one_to_three(2), 'II')

if __name__ == '__main__':
    unittest.main()
