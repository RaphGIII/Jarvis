def run(text: str) -> dict:
    lines = text.split("\n")
    non_empty_lines = [line for line in lines if line.strip() != ""]
    return {"line_count": len(non_empty_lines)}