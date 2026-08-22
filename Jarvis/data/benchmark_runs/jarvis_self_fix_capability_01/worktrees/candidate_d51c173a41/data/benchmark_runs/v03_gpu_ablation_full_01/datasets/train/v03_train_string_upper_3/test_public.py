import unittest
from solution import clean_upper

class PublicTests(unittest.TestCase):
    def test_public(self):
        self.assertEqual(clean_upper(' Ada '), 'ADA')

if __name__ == '__main__':
    unittest.main()
