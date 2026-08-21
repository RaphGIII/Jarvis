import unittest

import main


PUBLIC_CASES = [
  {
    "expected_output": {
      "most_common_value": 30
    },
    "input": {
      "column": "age",
      "csv_text": "name,age\nAlice,30\nBob,25\nAlice,25\nCharlie,30"
    }
  },
  {
    "expected_output": {
      "most_common_value": 3
    },
    "input": {
      "column": "count",
      "csv_text": "fruit,count\napple,5\nbanana,3\napple,7\norange,3"
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
