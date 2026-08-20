import unittest
from solution import parse_pairs

class PublicTests(unittest.TestCase):
    def test_public(self):
        self.assertEqual(parse_pairs('a=1,b=2'), {'a': '1', 'b': '2'})

if __name__ == '__main__':
    unittest.main()
