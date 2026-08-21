def parse_pairs(*args):
    text = args[0]
    return dict(part.split(':') for part in text.split(','))
