def countdown(*args):
    n = args[0]
    if n <= 0:
        return []
    return list(range(n, 0, -1))
