import unittest
from solution import nonempty_lines

class PublicTests(unittest.TestCase):
    def test_public(self):
        self.assertEqual(nonempty_lines('a\n\n b '), ['a', 'b'])

if __name__ == '__main__':
    unittest.main()
