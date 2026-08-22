def dedupe_keep_order(iterable):
    seen = set()
    result = []
    for value in iterable:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result
