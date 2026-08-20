import unittest
from solution import value

class PublicTests(unittest.TestCase):
    def test_value(self):
        self.assertEqual(value(), 99)

if __name__ == '__main__':
    unittest.main()
