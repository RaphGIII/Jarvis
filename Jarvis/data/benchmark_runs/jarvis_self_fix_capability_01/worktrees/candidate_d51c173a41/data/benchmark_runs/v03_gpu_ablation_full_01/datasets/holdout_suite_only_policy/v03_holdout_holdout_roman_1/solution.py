def roman_one_to_three(*args):
    n = args[0]
    if n == 1:
        return 'I'
    elif n == 2:
        return 'II'
    elif n == 3:
        return 'III'
    else:
        return ''

# Add docstring for clarity
__doc__ = "Convert integers 1, 2, or 3 to their Roman numeral representation."

# Ensure no extra logic or side effects
assert roman_one_to_three(1) == 'I'
assert roman_one_to_three(2) == 'II'
assert roman_one_to_three(3) == 'III'
