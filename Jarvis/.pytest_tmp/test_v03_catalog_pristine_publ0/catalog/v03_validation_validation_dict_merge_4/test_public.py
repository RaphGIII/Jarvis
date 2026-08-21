import unittest
from solution import merge_counts

class PublicTests(unittest.TestCase):
    def test_public(self):
        self.assertEqual(merge_counts({'a': 1}, {'a': 2, 'b': 1}), {'a': 3, 'b': 1})

if __name__ == '__main__':
    unittest.main()
