def is_palindrome(*args):
    text = args[0].replace(" ", "").lower()
    return text == text[::-1] if text else True
