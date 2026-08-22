def most_common(*args):
    if not args or not args[0]:
        return None
    values = list(args[0])
    count = {}
    for val in values:
        count[val] = count.get(val, 0) + 1
    return max(count, key=count.get) if count else None