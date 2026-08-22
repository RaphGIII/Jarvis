def roman_one_to_three(*args):
    n = args[0]
    if n == 1:
        return 'I'
    elif n == 2:
        return 'II'
    elif n == 3:
        return 'III'
    else:
        return 'N/A'