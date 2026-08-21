def mean_or_zero(*args):
    values = list(args) if args else []
    if len(values) == 0:
        return 0
    return sum(values) / len(values)
