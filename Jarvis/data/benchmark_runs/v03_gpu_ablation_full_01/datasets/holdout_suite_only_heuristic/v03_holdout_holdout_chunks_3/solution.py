def chunk_pairs(*args):
    values = args[0]
    if len(values) % 2 != 0:
        values.append(None)
    return [values[i:i+2] for i in range(0, len(values), 2)]
