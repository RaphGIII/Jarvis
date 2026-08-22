def even_values(*args):
    values = args[0] if args else []
    return [value for value in values if isinstance(value, int) and value % 2 == 0]
