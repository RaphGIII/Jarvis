def parse_optional_int(*args):
    text = args[0]
    if text is None:
        return None
    if isinstance(text, str) and text.strip() == "":
        return None
    try:
        return int(text)
    except (ValueError, TypeError):
        return None
