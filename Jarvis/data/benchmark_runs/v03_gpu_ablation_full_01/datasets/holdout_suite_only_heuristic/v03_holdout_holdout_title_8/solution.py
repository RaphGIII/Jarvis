def title_words(*args):
    text = args[0] if args else ''
    return text.strip().title()
