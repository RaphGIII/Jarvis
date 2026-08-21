def title_words(*args):
    text = args[0]
    words = text.split()
    return ' '.join(word.strip().capitalize() for word in words if word.strip())
