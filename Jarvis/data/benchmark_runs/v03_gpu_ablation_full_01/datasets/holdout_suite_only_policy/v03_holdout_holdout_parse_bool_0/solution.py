def parse_bool(*args):
    text = args[0]
    if text.lower() in ('yes', 'true', '1', 'on'):
        return True
    elif text.lower() in ('no', 'false', '0', 'off'):
        return False
    else:
        raise ValueError(f'Invalid boolean string: {text}')
