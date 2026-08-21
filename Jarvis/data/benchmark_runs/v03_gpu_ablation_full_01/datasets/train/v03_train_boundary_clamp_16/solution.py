def clamp(*args):
    value, low, high = args
    if value < low:
        return low
    elif value > high:
        return high
    return value
