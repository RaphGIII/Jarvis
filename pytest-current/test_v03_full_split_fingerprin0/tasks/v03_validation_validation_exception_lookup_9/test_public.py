import unittest
from solution import lookup_default

class PublicTests(unittest.TestCase):
    def test_public(self):
        self.assertEqual(lookup_default({}, 'x', 7), 7)

if __name__ == '__main__':
    unittest.main()
