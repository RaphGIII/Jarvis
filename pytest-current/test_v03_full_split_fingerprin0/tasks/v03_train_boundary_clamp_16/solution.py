def clamp(*args):
    value, low, high = args
    if value < low:
        return high
    if value > high:
        return low
    return value
