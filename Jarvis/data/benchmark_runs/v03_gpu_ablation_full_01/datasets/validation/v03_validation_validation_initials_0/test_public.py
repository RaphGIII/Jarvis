import unittest
from solution import make_initials

class PublicTests(unittest.TestCase):
    def test_public(self):
        self.assertEqual(make_initials('Ada Lovelace'), 'AL')

if __name__ == '__main__':
    unittest.main()
