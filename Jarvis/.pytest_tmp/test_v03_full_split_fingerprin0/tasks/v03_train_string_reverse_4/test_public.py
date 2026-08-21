import unittest
from solution import reverse_text

class PublicTests(unittest.TestCase):
    def test_public(self):
        self.assertEqual(reverse_text('abc'), 'cba')

if __name__ == '__main__':
    unittest.main()
