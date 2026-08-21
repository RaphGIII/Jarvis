def most_common(*args):
    if not args or not args[0]:
        return None
    values = args[0]
    if not values:
        return None
    return max(set(values), key=values.count)
