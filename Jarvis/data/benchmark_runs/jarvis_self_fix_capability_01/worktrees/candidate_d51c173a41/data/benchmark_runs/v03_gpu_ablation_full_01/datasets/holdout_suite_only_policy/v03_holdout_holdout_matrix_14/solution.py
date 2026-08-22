def transpose(*args):
    matrix = args[0]
    if not matrix or (not matrix[0] and not matrix):
        return []
    rows = len(matrix)
    cols = len(matrix[0])
    return [[matrix[j][i] for j in range(cols)] for i in range(rows)]
