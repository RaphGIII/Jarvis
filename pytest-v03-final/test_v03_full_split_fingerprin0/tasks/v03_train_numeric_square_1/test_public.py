import unittest
from solution import square_plus_one

class PublicTests(unittest.TestCase):
    def test_public(self):
        self.assertEqual(square_plus_one(3), 10)

if __name__ == '__main__':
    unittest.main()
