def letter_grade(*args):
    score = args[0]
    if score > 60:
        return 'fail'
    return 'pass'
