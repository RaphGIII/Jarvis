def nested_get(*args):
    mapping, outer, inner, default = args
    if not mapping or outer not in mapping:
        return default
    inner_value = mapping[outer]
    if not inner_value or inner not in inner_value:
        return default
    return inner_value[inner]
