import numpy as np

def generate_affine_matrix(first_row):
    matrix = [first_row]
    for i in range(1, 8):
        matrix.append(np.roll(matrix[i-1], 1))
    return np.array(matrix)

def gf2_rank(matrix):
    mat = matrix.copy()
    rows, cols = mat.shape
    rank = 0
    col = 0

    for r in range(rows):
        while col < cols and not mat[r:, col].any():
            col += 1
        if col == cols:
            break
        pivot = r + np.argmax(mat[r:, col])
        mat[[r, pivot]] = mat[[pivot, r]]
        for i in range(rows):
            if i != r and mat[i, col]:
                mat[i] ^= mat[r]
        rank += 1
        col += 1
    return rank

def matrix_properties(matrix):
    ones = int(matrix.sum())
    zeros = 64 - ones
    return {
        "Rank": gf2_rank(matrix),
        "Ones Count": ones,
        "Zeros Count": zeros
    }

def index_to_first_row(index):
    return np.array(list(map(int, f"{index:08b}")))
