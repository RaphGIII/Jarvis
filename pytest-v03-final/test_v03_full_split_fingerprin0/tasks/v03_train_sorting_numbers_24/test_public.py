import unittest
from solution import sort_desc

class PublicTests(unittest.TestCase):
    def test_public(self):
        self.assertEqual(sort_desc([1, 3, 2]), [3, 2, 1])

if __name__ == '__main__':
    unittest.main()
