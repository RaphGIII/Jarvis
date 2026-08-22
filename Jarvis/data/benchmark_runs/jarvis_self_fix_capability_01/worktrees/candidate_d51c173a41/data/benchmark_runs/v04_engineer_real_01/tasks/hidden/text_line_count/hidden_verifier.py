import main

HIDDEN_CASES = [
  {
    "expected": {
      "lines": 0
    },
    "input": {
      "text": ""
    }
  },
  {
    "expected": {
      "lines": 3
    },
    "input": {
      "text": "x\ny\nz"
    }
  }
]

for case in HIDDEN_CASES:
    result = main.run(case.get("input", {}))
    assert result == case["expected"], (case.get("input"), result, case["expected"])
