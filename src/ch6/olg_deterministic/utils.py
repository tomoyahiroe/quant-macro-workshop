# モデルの実装に必要な関数を定義する

import numpy as np
from numba import njit

# maliar グリッドを生成する関数
@njit
def maliar_grid(a_min, a_max, N, theta):
    a_grid = np.empty(N)
    for i in range(1,N+1):
        a_grid[i-1] = a_min + (a_max - a_min)*((i-1)/(N-1))**theta
        
    return a_grid