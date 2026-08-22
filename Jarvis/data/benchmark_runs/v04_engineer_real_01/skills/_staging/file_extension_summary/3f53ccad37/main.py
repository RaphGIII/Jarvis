def run(payload: dict) -> dict:
    """
    Summarizes file extensions from a list of paths.
    
    Args:
        payload (dict): A dictionary containing a list of file paths under the key 'paths'.

    Returns:
        dict: A dictionary with file extensions as keys and their counts as values (case-insensitive).
    """
    # Initialize a dictionary to store extension counts
    extension_counts = {}
    
    # Validate input
    if not payload or 'paths' not in payload or not payload['paths']:
        return {}
    
    # Process each path in the list
    for path in payload['paths']:
        if not path:
            continue
        
        # Split the path by dot to extract extension
        parts = path.split('.')
        
        # Ensure there's at least one part after splitting
        if len(parts) < 2:
            continue
        
        # Extract the last part as extension and convert to lowercase
        extension = parts[-1].lower()
        
        # Increment count for this extension
        extension_counts[extension] = extension_counts.get(extension, 0) + 1
    
    return extension_counts