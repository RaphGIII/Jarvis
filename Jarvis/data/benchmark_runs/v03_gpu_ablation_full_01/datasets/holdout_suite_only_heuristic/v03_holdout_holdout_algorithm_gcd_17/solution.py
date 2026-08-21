def gcd(a, b):
    """Compute the greatest common divisor of two integers."""
    while b:
        a, b = b, a % b
    return a

# Example usage
# print(gcd(12, 8)) # Should output 4