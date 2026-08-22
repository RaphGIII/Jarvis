def run(payload: dict) -> dict:
    if not payload or not isinstance(payload, list):
        return {}

    extension_count = {}

    for path in payload:
        if not isinstance(path, str):
            continue

        # Split the path by dot and take the last part as extension
        parts = path.split('.')
        if len(parts) > 1:
            extension = parts[-1].lower()
            extension_count[extension] = extension_count.get(extension, 0) + 1

    return extension_count