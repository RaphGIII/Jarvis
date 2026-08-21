import csv
import io
import json
import re
from collections import Counter
from pathlib import PurePath

CAPABILITY_ID = "text.line_count"


def run(payload: dict) -> dict:
    if CAPABILITY_ID == "text.line_count":
        return {"lines": sum(1 for line in str(payload.get("text", "")).splitlines() if line.strip())}
    if CAPABILITY_ID == "data.csv_column_mode":
        reader = csv.DictReader(io.StringIO(str(payload.get("csv_text", ""))))
        column = payload.get("column")
        values = [row[column] for row in reader if column in row and row.get(column) not in (None, "")]
        counts = Counter(values)
        if not counts:
            return {"value": None, "frequency": 0}
        value, frequency = sorted(counts.items(), key=lambda item: (-item[1], str(item[0])))[0]
        return {"value": value, "frequency": frequency}
    if CAPABILITY_ID == "files.extension_summary":
        counts = Counter(PurePath(str(path)).suffix.lower() for path in payload.get("paths", []))
        return dict(sorted(counts.items()))
    if CAPABILITY_ID == "data.json_records_to_csv":
        output = io.StringIO()
        fields = list(payload.get("fields", []))
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\r\n")
        writer.writeheader()
        for record in payload.get("records", []):
            writer.writerow({field: record.get(field, "") for field in fields})
        return {"csv": output.getvalue()}
    if CAPABILITY_ID == "text.markdown_table":
        headers = [str(item) for item in payload.get("headers", [])]
        rows = payload.get("rows", [])
        lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        for row in rows:
            lines.append("| " + " | ".join(str(item) for item in row) + " |")
        return {"markdown": "\n".join(lines)}
    if CAPABILITY_ID == "text.duplicate_lines":
        seen = set()
        duplicates = []
        for line in str(payload.get("text", "")).splitlines():
            value = line.strip()
            if not value:
                continue
            if value in seen and value not in duplicates:
                duplicates.append(value)
            seen.add(value)
        return {"duplicates": duplicates}
    if CAPABILITY_ID == "files.normalize_names":
        names = []
        for name in payload.get("names", []):
            text = str(name).strip().lower()
            if "." in text:
                stem, ext = text.rsplit(".", 1)
                normalized = re.sub(r"[^a-z0-9]+", "_", stem).strip("_") + "." + re.sub(r"[^a-z0-9]+", "", ext)
            else:
                normalized = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
            names.append(normalized)
        return {"names": names}
    if CAPABILITY_ID == "logs.level_counts":
        counts = {level: 0 for level in ["DEBUG", "INFO", "WARNING", "ERROR"]}
        for line in payload.get("lines", []):
            match = re.match(r"\s*(debug|info|warning|error)\b", str(line), flags=re.I)
            if match:
                counts[match.group(1).upper()] += 1
        return counts
    if CAPABILITY_ID == "records.filter_equals":
        field = payload.get("field")
        value = payload.get("value")
        return {"records": [record for record in payload.get("records", []) if record.get(field) == value]}
    if CAPABILITY_ID == "local.kv_utility":
        store = {}
        results = []
        for operation in payload.get("operations", []):
            op = operation[0]
            key = operation[1] if len(operation) > 1 else None
            if op == "set":
                store[key] = operation[2] if len(operation) > 2 else None
            elif op == "get":
                results.append(store.get(key))
            elif op == "delete":
                store.pop(key, None)
        return {"store": store, "results": results}
    if CAPABILITY_ID == "text.rule_transform":
        text = str(payload.get("text", ""))
        rule = str(payload.get("rule", "")).lower()
        if rule == "upper":
            return {"text": text.upper()}
        if rule == "lower":
            return {"text": text.lower()}
        if rule == "title":
            return {"text": text.title()}
        if rule == "reverse":
            return {"text": text[::-1]}
        raise ValueError("unknown rule")
    if CAPABILITY_ID == "data.json_key_compare":
        left = set((payload.get("left") or {}).keys())
        right = set((payload.get("right") or {}).keys())
        return {"added": sorted(right - left), "removed": sorted(left - right), "common": sorted(left & right)}
    if CAPABILITY_ID == "numbers.aggregate":
        values = list(payload.get("values", []))
        if not values:
            return {"count": 0, "sum": 0, "mean": None, "min": None, "max": None}
        total = sum(values)
        return {"count": len(values), "sum": total, "mean": total / len(values), "min": min(values), "max": max(values)}
    if CAPABILITY_ID == "text.parse_key_values":
        values = {}
        for line in str(payload.get("text", "")).splitlines():
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if key:
                values[key] = value.strip()
        return {"values": values}
    if CAPABILITY_ID == "sets.unique_sorted":
        return {"values": sorted(set(payload.get("values", [])), key=lambda item: str(item))}
    raise ValueError(f"Unsupported capability: {CAPABILITY_ID}")
