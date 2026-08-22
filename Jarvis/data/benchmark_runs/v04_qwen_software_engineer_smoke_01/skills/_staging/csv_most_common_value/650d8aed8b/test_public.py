import unittest

import main


PUBLIC_CASES = [
  {
    "expected_output": {
      "most_common_value": 25
    },
    "input": {
      "column": "age",
      "csv_text": "name,age\nAlice,25\nBob,30\nAlice,30\nCharlie,25"
    }
  },
  {
    "expected_output": {
      "most_common_value": 75
    },
    "input": {
      "column": "temperature",
      "csv_text": "city,temperature\nNew York,75\nLos Angeles,80\nNew York,70\nChicago,75"
    }
  }
]


class PublicSkillTests(unittest.TestCase):
    def test_public_cases(self):
        for case in PUBLIC_CASES:
            payload = case.get("input", {})
            if case.get("raises"):
                with self.assertRaises(Exception):
                    main.run(payload)
                continue
            result = main.run(payload)
            if "expected" in case:
                self.assertEqual(result, case["expected"], case.get("name", "public case"))
            for key in case.get("expected_keys", []):
                self.assertIsInstance(result, dict)
                self.assertIn(key, result)


if __name__ == "__main__":
    unittest.main()
