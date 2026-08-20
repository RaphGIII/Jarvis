def merge_counts(*args):
    left, right = args
    merged = dict(left)
    merged.update(right)
    return merged
