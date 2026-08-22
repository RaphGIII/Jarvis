import unittest

import main


PUBLIC_CASES = [
  {
    "expected_output": {
      "jpg": 1,
      "pdf": 2,
      "py": 1,
      "zip": 1
    },
    "input": [
      "document.pdf",
      "image.jpg",
      "script.py",
      "archive.zip",
      "document.pdf"
    ]
  },
  {
    "expected_output": {
      "csv": 1,
      "txt": 2
    },
    "input": [
      "report.txt",
      "data.csv",
      "report.txt"
    ]
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
