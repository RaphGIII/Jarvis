def is_anagram(*args):
    if len(args) != 2:
        return False
    a, b = args
    a_clean = ''.join(c.lower() for c in a if c.isalpha())
    b_clean = ''.join(c.lower() for c in b if c.isalpha())
    return len(a_clean) == len(b_clean) and sorted(a_clean) == sorted(b_clean) and a_clean and b_clean and len(a_clean) > 0
