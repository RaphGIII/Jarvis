import importlib
import json
import pathlib

payload = json.loads(pathlib.Path("request.json").read_text(encoding="utf-8"))
module = importlib.import_module("main")
result = module.run(payload)
pathlib.Path("output.json").write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
