def moving_sum(*args):
    if not args or not args[0]:
        return []
    values = list(args[0])
    return [values[i] + values[i+1] for i in range(len(values) - 1)] if len(values) > 1 else []
