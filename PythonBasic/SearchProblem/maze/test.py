import numpy as np
def manhattanDistance(point1, point2):
    return np.sum(np.abs(np.array(point1) - np.array(point2)))
def heuristic(wall):
    n = wall.shape[0]
    m = wall.shape[1]
    heuristic_matrix = np.zeros(shape=(n, m))
    for i in range(n):
        for j in range(m):
            if wall[i][j]:
                heuristic_matrix[i][j] = manhattanDistance((0, 5), (i, j))
    print(heuristic_matrix)        
wall = [
    [False, False, False, False, False, True, False],  # Dòng 1
    [False, False, False, False, False, True, False],  # Dòng 2
    [False, False, False, False, True, True, False],   # Dòng 3
    [False, False, False, False, True, True, False],   # Dòng 4
    [True,  True,  True,  True,  True, False, False],  # Dòng 5
    [True,  False, False, False, False, False, False]  # Dòng 6
]
wall = np.array(wall)
heuristic(wall)
