def nested_get(*args):
    mapping, outer, inner, default = args
    if not mapping or outer not in mapping or (outer in mapping and inner not in mapping[outer]):
        return default
    return mapping[outer][inner]
