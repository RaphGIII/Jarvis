import unittest
from solution import absolute_delta

class PublicTests(unittest.TestCase):
    def test_public(self):
        self.assertEqual(absolute_delta(2, 7), 5)

if __name__ == '__main__':
    unittest.main()
