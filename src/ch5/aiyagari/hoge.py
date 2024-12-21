import numpy as np
import stationary_dist as sd
import stationary_dist_numba as sdn
from numba import njit
import time

aprime_grid = np.array([[20, -48], [32, 15], [144, 27]])
agrid = np.array([12, 24, 36])
sd0 = np.full((3, 2), 1/6)
Pmatrix = np.array([[0.9, 0.1], [0.1, 0.9]])

# 時間を計測
start = time.time()
sd.sd_iteration(sd0, aprime_grid, agrid, Pmatrix)
end = time.time()
print(f"not numba: {end - start}")