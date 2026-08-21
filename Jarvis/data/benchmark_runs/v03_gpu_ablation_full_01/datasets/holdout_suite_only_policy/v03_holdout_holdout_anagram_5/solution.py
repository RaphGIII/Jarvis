def is_anagram(*args):
    if len(args) != 2:
        return False
    a, b = args
    a_clean = a.lower().replace(' ', '')
    b_clean = b.lower().replace(' ', '')
    return sorted(a_clean) == sorted(b_clean) and a_clean.isalpha() and b_clean.isalpha() and len(a_clean) == len(b_clean)
