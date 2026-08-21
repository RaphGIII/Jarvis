import unittest
from solution import dedupe_keep_order

class PublicTests(unittest.TestCase):
    def test_public(self):
        self.assertEqual(dedupe_keep_order([2, 1, 2]), [2, 1])

if __name__ == '__main__':
    unittest.main()
