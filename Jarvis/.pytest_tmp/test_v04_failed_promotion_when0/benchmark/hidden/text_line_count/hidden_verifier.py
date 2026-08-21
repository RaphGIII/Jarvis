import main

HIDDEN_CASES = [
  {
    "expected": {
      "wrong": true
    },
    "input": {
      "text": "one\n\nthree"
    }
  }
]

for case in HIDDEN_CASES:
    result = main.run(case.get("input", {}))
    assert result == case["expected"], (case.get("input"), result, case["expected"])
