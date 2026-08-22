def sum_even(*args):
    values = args[0]
    total = 0
    for v in values:
        if v % 2 == 0:
            total += v
    return total
