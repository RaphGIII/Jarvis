def unique_sorted(*args):
    values = []
    for arg in args:
        values.extend(arg)
    return sorted(set(values))