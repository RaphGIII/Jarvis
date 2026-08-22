import unittest
from solution import safe_get

class PublicTests(unittest.TestCase):
    def test_public(self):
        self.assertEqual(safe_get([1], 5, None), None)

if __name__ == '__main__':
    unittest.main()
