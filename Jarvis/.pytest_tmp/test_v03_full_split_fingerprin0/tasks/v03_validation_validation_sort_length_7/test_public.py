import unittest
from solution import sort_by_length

class PublicTests(unittest.TestCase):
    def test_public(self):
        self.assertEqual(sort_by_length(['bbb', 'a', 'cc']), ['a', 'cc', 'bbb'])

if __name__ == '__main__':
    unittest.main()
