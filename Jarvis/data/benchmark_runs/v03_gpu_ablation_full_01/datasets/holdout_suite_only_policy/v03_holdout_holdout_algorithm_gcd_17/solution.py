def gcd(*args):
    if len(args) == 0:
        return 0
    a, b = args[0], args[1]
    while b:
        a, b = b, a % b
    return a

# Extended support for multiple arguments
def gcd_multiple(*numbers):
    result = numbers[0]
    for num in numbers[1:]:
        result = gcd(result, num)
    return result


