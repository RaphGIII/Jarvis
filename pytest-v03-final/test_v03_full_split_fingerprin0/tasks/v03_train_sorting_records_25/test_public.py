import unittest
from solution import sort_by_name

class PublicTests(unittest.TestCase):
    def test_public(self):
        self.assertEqual(sort_by_name([{'name':'b'}, {'name':'a'}]), [{'name':'a'}, {'name':'b'}])

if __name__ == '__main__':
    unittest.main()
