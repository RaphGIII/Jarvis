def count_words(*args):
    if not args:
        return {}
    words = args[0]
    return {word: 1 for word in words}
