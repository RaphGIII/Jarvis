def nested_get(*args):
    mapping, outer, inner, default = args
    if outer not in mapping:
        return default
    if inner not in mapping.get(outer, {}):
        return default
    return mapping[outer][inner]