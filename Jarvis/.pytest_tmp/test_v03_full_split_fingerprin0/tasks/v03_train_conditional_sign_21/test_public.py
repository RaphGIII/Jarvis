import unittest
from solution import sign_label

class PublicTests(unittest.TestCase):
    def test_public(self):
        self.assertEqual(sign_label(0), 'zero')

if __name__ == '__main__':
    unittest.main()
