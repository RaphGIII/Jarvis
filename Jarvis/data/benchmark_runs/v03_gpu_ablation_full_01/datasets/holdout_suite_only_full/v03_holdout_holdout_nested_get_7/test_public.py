import unittest
from solution import nested_get

class PublicTests(unittest.TestCase):
    def test_public(self):
        self.assertEqual(nested_get({}, 'a', 'b', 9), 9)

if __name__ == '__main__':
    unittest.main()
