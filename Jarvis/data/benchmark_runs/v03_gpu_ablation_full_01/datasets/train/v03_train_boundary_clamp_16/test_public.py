import unittest
from solution import clamp

class PublicTests(unittest.TestCase):
    def test_public(self):
        self.assertEqual(clamp(-1, 0, 10), 0)

if __name__ == '__main__':
    unittest.main()
