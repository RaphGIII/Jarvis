def chunk_pairs(*args):
    values = list(args[0]) if args[0] else []
    if len(values) % 2 != 0:
        values.append(None)
    return [list(pair) for pair in [[values[i], values[i+1]] for i in range(0, len(values), 2)]]
