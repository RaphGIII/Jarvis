import unittest

import main


INTERNAL_CASES = [{'name': 'valid_csv_with_single_column', 'payload': {'column': 'age', 'csv_text': 'name,age\nAlice,25\nBob,30\nAlice,30\nCharlie,25'}, 'expected': {'most_common_value': 25}, 'raises': False, 'invariant': None}, {'name': 'valid_csv_with_different_column', 'payload': {'column': 'temperature', 'csv_text': 'city,temperature\nNew York,75\nLos Angeles,80\nNew York,70\nChicago,75'}, 'expected': {'most_common_value': 75}, 'raises': False, 'invariant': None}, {'name': 'empty_csv', 'payload': {'column': 'age', 'csv_text': 'name,age\n\n\n'}, 'expected': {'most_common_value': None}, 'raises': False, 'invariant': None}, {'name': 'single_row_csv', 'payload': {'column': 'age', 'csv_text': 'name,age\nAlice,25'}, 'expected': {'most_common_value': 25}, 'raises': False, 'invariant': None}, {'name': 'all_same_values', 'payload': {'column': 'age', 'csv_text': 'name,age\nAlice,25\nBob,25\nCharlie,25'}, 'expected': {'most_common_value': 25}, 'raises': False, 'invariant': None}, {'name': 'column_not_in_csv', 'payload': {'column': 'salary', 'csv_text': 'name,age\nAlice,25\nBob,30'}, 'expected': {'most_common_value': None}, 'raises': False, 'invariant': None}, {'name': 'no_data_in_column', 'payload': {'column': 'age', 'csv_text': 'name,age\nAlice,,\nBob,30\nCharlie,'}, 'expected': {'most_common_value': None}, 'raises': False, 'invariant': None}, {'name': 'duplicate_header', 'payload': {'column': 'age', 'csv_text': 'name,age\nname,age\nAlice,25\nBob,30'}, 'expected': {'most_common_value': 25}, 'raises': False, 'invariant': None}, {'name': 'public_keys', 'payload': {'csv_text': 'name,age\nAlice,25\nBob,30\nAlice,30\nCharlie,25', 'column': 'age'}, 'expected': None, 'raises': False, 'invariant': 'returns_dict'}, {'name': 'public_keys', 'payload': {'csv_text': 'city,temperature\nNew York,75\nLos Angeles,80\nNew York,70\nChicago,75', 'column': 'temperature'}, 'expected': None, 'raises': False, 'invariant': 'returns_dict'}]


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
