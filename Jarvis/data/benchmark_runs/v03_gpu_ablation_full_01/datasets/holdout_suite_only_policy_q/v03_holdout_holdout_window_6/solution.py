def moving_sum(*args):
    values = list(args[0]) if args[0] else []
    if len(values) < 2:
        return []
    return [values[i] + values[i+1] for i in range(len(values) - 1)]
