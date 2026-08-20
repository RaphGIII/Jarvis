import unittest
from solution import group_by_first

class PublicTests(unittest.TestCase):
    def test_public(self):
        self.assertEqual(group_by_first(['ant', 'bat', 'ape']), {'a': ['ant', 'ape'], 'b': ['bat']})

if __name__ == '__main__':
    unittest.main()
