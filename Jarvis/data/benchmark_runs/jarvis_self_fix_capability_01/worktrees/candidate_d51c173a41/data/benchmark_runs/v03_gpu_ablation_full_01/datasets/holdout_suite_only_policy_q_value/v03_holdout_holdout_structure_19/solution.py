def group_by_first(*args):
    if not args or not args[0]:
        return {}
    grouped = {}
    for word in args[0]:
        key = word[0].lower()
        grouped.setdefault(key, []).append(word)
    return grouped
