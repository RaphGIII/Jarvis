import unittest
from solution import parse_bool

class PublicTests(unittest.TestCase):
    def test_public(self):
        self.assertTrue(parse_bool('yes'))

if __name__ == '__main__':
    unittest.main()
