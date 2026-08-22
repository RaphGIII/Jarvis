import unittest

import main


INTERNAL_CASES = [{'name': 'empty_input', 'payload': {'text': ''}, 'expected': {'line_count': 0}, 'raises': False, 'invariant': None}, {'name': 'input_with_only_whitespace', 'payload': {'text': '   \n\n\t '}, 'expected': {'line_count': 0}, 'raises': False, 'invariant': None}, {'name': 'input_with_non_empty_lines', 'payload': {'text': 'line1\n  \nline2\n\nline3'}, 'expected': {'line_count': 3}, 'raises': False, 'invariant': None}, {'name': 'single_non_empty_line', 'payload': {'text': 'a\nb\nc\n'}, 'expected': {'line_count': 3}, 'raises': False, 'invariant': None}, {'name': 'all_empty_lines', 'payload': {'text': '\n\n\n'}, 'expected': {'line_count': 0}, 'raises': False, 'invariant': None}, {'name': 'mixed_whitespace_and_content', 'payload': {'text': '  \nhello\n  \nworld  \n'}, 'expected': {'line_count': 2}, 'raises': False, 'invariant': None}, {'name': 'trailing_newline_with_non_empty_line', 'payload': {'text': 'a\n\nb\n'}, 'expected': {'line_count': 2}, 'raises': False, 'invariant': None}, {'name': 'mixed_line_endings', 'payload': {'text': 'line1\r\nline2\r\n\nline3'}, 'expected': {'line_count': 3}, 'raises': False, 'invariant': None}, {'name': 'single_line_with_only_whitespace', 'payload': {'text': '   '}, 'expected': {'line_count': 0}, 'raises': False, 'invariant': None}, {'name': 'empty_string_with_no_whitespace', 'payload': {'text': ''}, 'expected': {'line_count': 0}, 'raises': False, 'invariant': None}, {'name': 'non_ascii_whitespace_characters', 'payload': {'text': 'a\u200b\nb\u200c\nc'}, 'expected': {'line_count': 3}, 'raises': False, 'invariant': None}, {'name': 'mixed_line_endings_with_empty_lines', 'payload': {'text': 'line1\r\n\r\nline2\r\n\nline3'}, 'expected': {'line_count': 3}, 'raises': False, 'invariant': None}]


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
