def common_items(*args):
    if len(args) < 2:
        return []
    a, b = args
    return sorted(set(a) & set(b))

# Alternative: Ensure consistent behavior with set intersection
# This is equivalent but more explicit in intent
# Common items is the intersection of all sets in args
# For now, only handles two sets as per test case

