def invert_lookup(*args):
    mapping = args[0]
    return {v: k for k, v in mapping.items()}