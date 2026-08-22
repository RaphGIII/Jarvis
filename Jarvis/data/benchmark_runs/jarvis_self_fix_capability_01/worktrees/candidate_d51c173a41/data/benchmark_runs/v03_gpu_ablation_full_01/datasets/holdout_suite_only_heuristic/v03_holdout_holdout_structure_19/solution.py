def group_by_first(*args):
    words = args[0]
    result = {}
    for word in words:
        first_letter = word[0]
        if first_letter not in result:
            result[first_letter] = []
        result[first_letter].append(word)
    return result
