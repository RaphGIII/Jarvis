def run(payload: dict) -> dict:
    if isinstance(payload, str):
        text = payload
    else:
        text = payload.get("text", "")
    lines = text.split("\n")
    non_empty_count = 0
    for line in lines:
        stripped = line.strip()
        if stripped != "":
            non_empty_count += 1
    return {"line_count": non_empty_count}