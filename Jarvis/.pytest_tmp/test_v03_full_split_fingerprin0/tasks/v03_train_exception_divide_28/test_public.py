import unittest
from solution import safe_divide

class PublicTests(unittest.TestCase):
    def test_public(self):
        self.assertIsNone(safe_divide(4, 0))

if __name__ == '__main__':
    unittest.main()
