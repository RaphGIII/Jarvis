def make_initials(*args):
    name = args[0]
    if not name or not name.strip():
        return ''
    parts = name.split()
    return ''.join(part[0].upper() for part in parts if part.strip())
