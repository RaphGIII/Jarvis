def group_by_first(*args):
    words = args[0] if args else []
    grouped = {}
    for word in words:
        first_letter = word[0].lower()
        if first_letter not in grouped:
            grouped[first_letter] = []
        grouped[first_letter].append(word.lower())
    return grouped
