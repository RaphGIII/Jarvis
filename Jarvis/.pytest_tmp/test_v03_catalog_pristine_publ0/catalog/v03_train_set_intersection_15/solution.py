def common_items(*args):
    a, b = args
    return sorted(set(a) | set(b))
