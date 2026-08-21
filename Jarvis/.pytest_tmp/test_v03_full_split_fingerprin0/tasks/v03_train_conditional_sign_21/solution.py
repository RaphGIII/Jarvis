def sign_label(*args):
    n = args[0]
    if n >= 0:
        return 'positive'
    return 'negative'
