def parse_optional_int(*args):
    if not args:
        return None
    text = args[0]
    if text is None or not isinstance(text, str):
        return None
    try:
        return int(text)
    except ValueError:
        return None
