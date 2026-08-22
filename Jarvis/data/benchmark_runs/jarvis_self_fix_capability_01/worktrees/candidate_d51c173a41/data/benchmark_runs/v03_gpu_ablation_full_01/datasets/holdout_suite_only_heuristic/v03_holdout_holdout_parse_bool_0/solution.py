def parse_bool(*args):
    text = args[0]
    if text.lower() in ['yes', 'true', '1', 'y', 't']:
        return True
    elif text.lower() in ['no', 'false', '0', 'n', 'f']:
        return False
    else:
        return False