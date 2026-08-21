def moving_sum(*args):
    values = args[0] if args else []
    if not values:
        return []
    result = []
    for i in range(len(values) - 1):
        result.append(values[i] + values[i + 1])
    return result