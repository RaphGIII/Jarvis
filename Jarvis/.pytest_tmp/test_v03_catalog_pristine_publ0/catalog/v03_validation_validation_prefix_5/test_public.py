import unittest
from solution import starts_with_any

class PublicTests(unittest.TestCase):
    def test_public(self):
        self.assertTrue(starts_with_any('jarvis', ['ja', 'co']))

if __name__ == '__main__':
    unittest.main()
