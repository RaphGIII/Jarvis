import unittest

import main


INTERNAL_CASES = [{'name': 'Empty input list', 'payload': {'paths': []}, 'expected': {'jpg': 0, 'pdf': 0, 'py': 0, 'zip': 0}, 'raises': False, 'invariant': None}, {'name': 'Single file with unique extension', 'payload': {'paths': ['example.txt']}, 'expected': {'txt': 1}, 'raises': False, 'invariant': None}, {'name': 'Multiple files with same extension (case-sensitive)', 'payload': {'paths': ['File.PDF', 'file.pdf', 'FILE.pdf']}, 'expected': {'pdf': 3}, 'raises': False, 'invariant': None}, {'name': 'All files with no extension', 'payload': {'paths': ['noext', 'file.with.many.extensions']}, 'expected': {'': '0'}, 'raises': False, 'invariant': None}, {'name': 'Mixed case extensions', 'payload': {'paths': ['Image.JPEG', 'Data.PDF', 'script.PY']}, 'expected': {'jpeg': 1, 'pdf': 1, 'py': 1}, 'raises': False, 'invariant': None}, {'name': 'Only one extension type', 'payload': {'paths': ['test.py', 'test.py', 'test.py']}, 'expected': {'py': 3}, 'raises': False, 'invariant': None}, {'name': 'Empty string in path list', 'payload': {'paths': ['', 'file.txt']}, 'expected': {'txt': 1}, 'raises': False, 'invariant': None}]


class InternalQATests(unittest.TestCase):
    def test_contract_cases(self):
        for case in INTERNAL_CASES:
            payload = case.get("payload", {})
            if case.get("raises"):
                with self.assertRaises(Exception, msg=case.get("name", "raises")):
                    main.run(payload)
                continue
            result = main.run(payload)
            self.assertIsInstance(result, dict, case.get("name", "returns dict"))
            if case.get("expected") is not None:
                self.assertEqual(result, case["expected"], case.get("name", "internal case"))



if __name__ == "__main__":
    unittest.main()
