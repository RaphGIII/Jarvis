def most_common(*args):
    values = list(args[0]) if args else []
    count = {}
    for val in values:
        count[val] = count.get(val, 0) + 1
    return max(count, key=lambda x: count[x]) if count else None

