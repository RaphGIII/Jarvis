import unittest
from solution import title_words

class PublicTests(unittest.TestCase):
    def test_public(self):
        self.assertEqual(title_words('hello world'), 'Hello World')

if __name__ == '__main__':
    unittest.main()
