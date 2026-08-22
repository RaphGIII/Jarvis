import unittest
from solution import count_words

class PublicTests(unittest.TestCase):
    def test_public(self):
        self.assertEqual(count_words(['a', 'a', 'b']), {'a': 2, 'b': 1})

if __name__ == '__main__':
    unittest.main()
