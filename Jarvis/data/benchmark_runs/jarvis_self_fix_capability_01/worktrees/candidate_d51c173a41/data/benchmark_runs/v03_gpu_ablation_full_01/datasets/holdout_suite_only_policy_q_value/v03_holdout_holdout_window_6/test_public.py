import unittest
from solution import moving_sum

class PublicTests(unittest.TestCase):
    def test_public(self):
        self.assertEqual(moving_sum([1, 2, 3]), [3, 5])

if __name__ == '__main__':
    unittest.main()
