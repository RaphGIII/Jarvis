import unittest
from solution import is_valid_port

class PublicTests(unittest.TestCase):
    def test_public(self):
        self.assertFalse(is_valid_port(70000))

if __name__ == '__main__':
    unittest.main()
