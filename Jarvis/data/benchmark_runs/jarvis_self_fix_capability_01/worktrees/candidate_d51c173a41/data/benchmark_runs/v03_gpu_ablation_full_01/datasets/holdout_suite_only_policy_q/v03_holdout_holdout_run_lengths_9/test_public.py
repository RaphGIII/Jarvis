import unittest
from solution import run_lengths

class PublicTests(unittest.TestCase):
    def test_public(self):
        self.assertEqual(run_lengths('aabb'), [('a', 2), ('b', 2)])

if __name__ == '__main__':
    unittest.main()
