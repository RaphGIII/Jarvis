import unittest
from solution import contains_casefold

class PublicTests(unittest.TestCase):
    def test_public(self):
        self.assertTrue(contains_casefold('Hello', 'he'))

if __name__ == '__main__':
    unittest.main()
