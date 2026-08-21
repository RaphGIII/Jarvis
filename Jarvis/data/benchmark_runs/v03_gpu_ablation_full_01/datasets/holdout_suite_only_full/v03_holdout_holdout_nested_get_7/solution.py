def nested_get(*args):
    mapping, outer, inner, default = args
    current = mapping
    try:
        current = current[outer]
        return current[inner]
    except (TypeError, KeyError):
        return default
    return default
