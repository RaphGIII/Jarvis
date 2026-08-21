def rotate_left(values, n):
    if n <= 0 or not values:
        return values.copy()
    n = n % len(values)
    return values[n:] + values[:n]
