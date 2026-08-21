import unittest
from solution import countdown

class PublicTests(unittest.TestCase):
    def test_public(self):
        self.assertEqual(countdown(3), [3, 2, 1])

if __name__ == '__main__':
    unittest.main()
