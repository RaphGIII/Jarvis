import unittest
from solution import first_index

class PublicTests(unittest.TestCase):
    def test_public(self):
        self.assertEqual(first_index([1, 2], 3), -1)

if __name__ == '__main__':
    unittest.main()
