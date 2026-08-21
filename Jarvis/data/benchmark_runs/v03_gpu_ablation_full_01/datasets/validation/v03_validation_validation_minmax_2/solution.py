def min_max(*args):
    values = list(args)
    if not values:
        return None
    return (min(values), max(values))
