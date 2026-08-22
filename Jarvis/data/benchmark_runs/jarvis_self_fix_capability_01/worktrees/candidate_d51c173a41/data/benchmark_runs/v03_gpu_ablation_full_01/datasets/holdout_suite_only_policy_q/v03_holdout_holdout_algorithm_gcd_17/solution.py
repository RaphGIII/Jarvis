def gcd(*args):
    if len(args) == 0:
        return 0
    a, b = args[0], args[1]
    while b:
        a, b = b, a % b
    return a
