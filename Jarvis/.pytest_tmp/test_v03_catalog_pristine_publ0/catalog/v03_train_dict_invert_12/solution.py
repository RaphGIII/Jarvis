def invert_lookup(*args):
    mapping = args[0]
    return {k: v for k, v in mapping.items()}
