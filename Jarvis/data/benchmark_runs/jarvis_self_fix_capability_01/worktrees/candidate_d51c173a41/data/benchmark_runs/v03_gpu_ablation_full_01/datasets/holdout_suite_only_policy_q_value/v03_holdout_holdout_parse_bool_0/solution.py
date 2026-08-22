def parse_bool(*args):
    text = args[0]
    if text.lower() in ('yes', 'true', '1', 'on', 'y', 't', 'yes'):
        return True
    elif text.lower() in ('no', 'false', '0', 'off', 'n', 'f', 'no'):
        return False
    else:
        raise ValueError(f'Invalid boolean string: {text}')
