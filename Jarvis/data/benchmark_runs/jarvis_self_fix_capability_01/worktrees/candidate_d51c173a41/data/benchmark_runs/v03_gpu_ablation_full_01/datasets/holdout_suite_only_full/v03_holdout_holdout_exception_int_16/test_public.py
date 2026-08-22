import unittest
from solution import parse_optional_int

class PublicTests(unittest.TestCase):
    def test_public(self):
        self.assertIsNone(parse_optional_int('x'))

if __name__ == '__main__':
    unittest.main()
