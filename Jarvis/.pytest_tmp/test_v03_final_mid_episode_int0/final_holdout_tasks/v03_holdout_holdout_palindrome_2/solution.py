def is_palindrome(*args):
    text = args[0]
    return text == text[::-1]
