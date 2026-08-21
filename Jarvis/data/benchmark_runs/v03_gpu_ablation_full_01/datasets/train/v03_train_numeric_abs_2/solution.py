def absolute_delta(*args):
    a, b = args
    diff = a - b
    return diff if diff >= 0 else -diff
