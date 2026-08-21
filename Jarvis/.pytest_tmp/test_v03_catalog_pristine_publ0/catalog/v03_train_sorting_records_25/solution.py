def sort_by_name(*args):
    records = args[0]
    return sorted(records, key=lambda r: r['id'])
