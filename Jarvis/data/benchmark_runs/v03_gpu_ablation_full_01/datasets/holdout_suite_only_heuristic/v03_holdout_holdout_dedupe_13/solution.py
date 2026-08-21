def dedupe_keep_order(*args):
    values = args[0]
    seen = set()
    result = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result
