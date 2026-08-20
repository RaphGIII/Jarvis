import unittest
from solution import parse_ints

class PublicTests(unittest.TestCase):
    def test_public(self):
        self.assertEqual(parse_ints('1,2,3'), [1, 2, 3])

if __name__ == '__main__':
    unittest.main()
