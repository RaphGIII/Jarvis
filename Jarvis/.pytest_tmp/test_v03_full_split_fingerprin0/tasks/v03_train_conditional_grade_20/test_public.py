import unittest
from solution import letter_grade

class PublicTests(unittest.TestCase):
    def test_public(self):
        self.assertEqual(letter_grade(80), 'pass')

if __name__ == '__main__':
    unittest.main()
