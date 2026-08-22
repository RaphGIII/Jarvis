def safe_get(*args):
    values, index, default = args
    try:
        return values[index]
    except IndexError:
        return default
