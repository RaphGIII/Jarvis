def roman_one_to_three(n):
    roman_map = {1: 'I', 2: 'II', 3: 'III'}
    return roman_map[n] if n in roman_map else ''

