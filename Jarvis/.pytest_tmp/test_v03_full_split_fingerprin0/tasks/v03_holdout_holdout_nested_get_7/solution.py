def nested_get(*args):
    mapping, outer, inner, default = args
    return mapping[outer][inner]
