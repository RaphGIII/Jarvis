import main

HIDDEN_CASES = [
  {
    "expected": {
      "frequency": 2,
      "value": "y"
    },
    "input": {
      "column": "b",
      "csv_text": "a,b\n1,x\n2,y\n3,y\n"
    }
  }
]

for case in HIDDEN_CASES:
    result = main.run(case.get("input", {}))
    assert result == case["expected"], (case.get("input"), result, case["expected"])
