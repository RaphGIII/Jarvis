def is_valid_port(value):
    if not isinstance(value, int):
        return False
    if value < 1 or value > 65535:
        return False
    return True

# Ensure port validation handles strings and floats correctly by checking type before comparison
