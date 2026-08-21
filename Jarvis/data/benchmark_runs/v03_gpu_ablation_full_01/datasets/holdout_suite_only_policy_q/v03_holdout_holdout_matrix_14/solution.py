def transpose(matrix):
    if not matrix or not matrix[0]:
        return []
    rows = len(matrix)
    cols = len(matrix[0])
    return [[matrix[j][i] for j in range(cols)] for i in range(rows)]