def make_initials(*args):
    name = args[0]
    return ''.join(part[-1].lower() for part in name.split())
