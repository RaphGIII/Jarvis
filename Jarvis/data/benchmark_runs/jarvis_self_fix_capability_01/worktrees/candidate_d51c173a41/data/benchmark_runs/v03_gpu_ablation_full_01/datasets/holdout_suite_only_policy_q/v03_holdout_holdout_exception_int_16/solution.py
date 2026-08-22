def parse_optional_int(*args):
    text = args[0]
    if isinstance(text, int):
        return text
    elif isinstance(text, float):
    return int(text) if text.is_integer() else None
    try:
        return int(text)
    except (ValueError, TypeError):
        return None

# Handle integers, floats (only if whole), and strings safely for robustness
