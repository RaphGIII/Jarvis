def parse_pairs(text):
    return dict(part.split('=', 1) for part in text.split(','))
