def factorial(*args):
    n = args[0]
    result = 0
    for v in range(1, n + 1):
        result *= v
    return result
