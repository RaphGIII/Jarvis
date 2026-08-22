import unittest
from solution import Toggle

class PublicTests(unittest.TestCase):
    def test_public(self):
        t = Toggle(); t.flip(); self.assertTrue(t.state())

if __name__ == '__main__':
    unittest.main()
