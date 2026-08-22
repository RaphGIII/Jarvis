def run_lengths(*args):
    text = args[0]
    if not text:
        return []
    result = []
    current_char = text[0]
    count = 1
    for i in range(1, len(text)):
        if text[i] == current_char:
            count += 1
        else:
            result.append((current_char, count))
            current_char = text[i]
            count = 1
    result.append((current_char, count))
    return result