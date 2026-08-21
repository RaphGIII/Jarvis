import unittest
from solution import min_max

class PublicTests(unittest.TestCase):
    def test_public(self):
        self.assertEqual(min_max([3, 1, 4]), (1, 4))

if __name__ == '__main__':
    unittest.main()
