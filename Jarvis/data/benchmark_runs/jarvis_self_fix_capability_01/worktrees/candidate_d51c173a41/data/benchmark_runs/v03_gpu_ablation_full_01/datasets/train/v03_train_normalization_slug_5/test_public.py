import unittest
from solution import slugify

class PublicTests(unittest.TestCase):
    def test_public(self):
        self.assertEqual(slugify('Hello World'), 'hello-world')

if __name__ == '__main__':
    unittest.main()
