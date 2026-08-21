import unittest
from solution import common_items

class PublicTests(unittest.TestCase):
    def test_public(self):
        self.assertEqual(common_items([1, 2], [2, 3]), [2])

if __name__ == '__main__':
    unittest.main()
