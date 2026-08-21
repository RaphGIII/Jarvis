import unittest
from calculator import add

class CalculatorTests(unittest.TestCase):
    def test_adds(self):
        self.assertEqual(add(2, 3), 5)

if __name__ == '__main__':
    unittest.main()
