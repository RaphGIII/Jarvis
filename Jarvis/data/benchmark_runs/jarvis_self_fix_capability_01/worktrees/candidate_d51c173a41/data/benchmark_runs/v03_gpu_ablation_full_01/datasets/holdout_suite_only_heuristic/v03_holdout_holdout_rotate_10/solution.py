def rotate_left(values, n):
    if not values or n <= 0:
        return values.copy()
    n = n % len(values)
    return values[n:] + values[:n]