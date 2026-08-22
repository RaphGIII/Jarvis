import main

HIDDEN_CASES = [
  {
    "expected": {
      "": 1,
      ".json": 2
    },
    "input": {
      "paths": [
        "x.JSON",
        "dir/y.json",
        "z"
      ]
    }
  }
]

for case in HIDDEN_CASES:
    result = main.run(case.get("input", {}))
    assert result == case["expected"], (case.get("input"), result, case["expected"])
