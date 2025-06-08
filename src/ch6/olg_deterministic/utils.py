"""モデルに関係なく再利用するような関数を定義する
モデルに依存せず再利用可能なユーティリティ関数を定義するモジュール。
functions:
    maliar_grid(a_min, a_max, N, theta):
        Maliar et al. （2010）の方法を用いて、a_minからa_maxまでの範囲で非線形に間隔を取ったグリッドを生成する関数。
        thetaパラメータによりグリッドの密度を調整できる。
"""
import numpy as np
from numba import njit

# maliar グリッドを生成する関数
@njit
def maliar_grid(a_min: float, a_max: float, N: int, theta: float) -> np.ndarray:
    """_summary_

    Args:
        a_min (float): グリッドの最小値
        a_max (float): グリッドの最大値
        N (int):  グリッドの点の数
        theta (float):  グリッドの非線形性を制御するパラメータ

    Returns:
        np.ndarray: 生成されたグリッドの配列
    生成されたグリッドは、a_minからa_maxまでの範囲で、非線形に間隔を取ったN個の点を含む。
    """
    a_grid = np.empty(N)
    for i in range(1, N + 1):
        a_grid[i - 1] = a_min + (a_max - a_min) * ((i - 1) / (N - 1)) ** theta

    return a_grid