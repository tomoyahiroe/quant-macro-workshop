import numpy as np
from typing import Callable
from numba import njit

# TODO: 定常分布が政策関数と同じ要素数であることを前提としてしまっている

# 政策関数によって得られたグリッド外の値が、グリッド上に移動する確率行列を計算する関数
@njit
def split_prob_to_agrid(aprime: float, agrid: np.ndarray) -> np.ndarray:
    """ 政策関数によって得られた a の値が、a_grid のどの点に移動するのかを示す確率ベクトルを計算する関数

    Args:
    aprime: 政策関数によって得られた a の値
    agrid: a のグリッド
    return: aprime が agrid のどの点に移動するのかを示す確率ベクトル
    """
    probs = np.zeros(len(agrid))
    # a の値が a_grid の最小値より小さい場合
    if aprime <= agrid[0]:
        probs[0] = 1.0
        return probs

    # a の値が a_grid の最大値より大きい場合
    if aprime >= agrid[-1]:
        probs[-1] = 1.0
        return probs

    # a の値が a_grid の最小値より大きく最大値より小さい場合
    for i in range(len(agrid)-1):
        if agrid[i] <= aprime <= agrid[i+1]:
            probs[i] = (agrid[i+1] - aprime) / (agrid[i+1] - agrid[i])
            probs[i+1] = (aprime - agrid[i]) / (agrid[i+1] - agrid[i])
            return probs

    # この行までに return されていない場合はエラーを出す
    raise ValueError("Splitting probability failed.")

@njit
def calc_weight_matrix(aprime_grid: np.ndarray, agrid: np.ndarray, split_prob_to_agrid = split_prob_to_agrid) -> np.ndarray:
    """ 前期の定常分布に係る重みづけ確率行列 weight_matrix を計算する関数

    Args:
        aprime_grid: 政策関数の行き先のグリッド
        agrid: 資産のグリッド
        split_prob: 政策関数によって得られた a の値が、a_grid のどの点に移動するのかを示す確率ベクトルを計算する関数

    Returns:
        weight_matrix: 前期の定常分布に係る重みづけ確率のグリッド、shape は aprime_grid と同じ
    """
    weight_matrix = np.empty((len(aprime_grid), len(aprime_grid[0]), len(agrid)))

    # aprime_grid は 二次元配列
    for i in range(len(aprime_grid)):
        for j in range(len(aprime_grid[0])):
            weight_matrix[i, j] = split_prob_to_agrid(float(aprime_grid[i, j]), agrid)
    return weight_matrix

@njit
def gen_pmesh(aprime_grid: np.ndarray, P: np.ndarray, next_y_index: int) -> np.ndarray:
    """ 遷移確率を 政策関数 aprime_grid との要素ごとの計算に使えるように meshgrid を生成する関数

    Args:
        aprime_grid (np.ndarray): 政策関数のグリッド
        P (np.ndarray): 遷移確率行列
        next_y_index (int): 次期の所得状態 y' のインデックス

    Returns:
        np.ndarray: 政策関数と遷移確率行列を使って計算するための meshgrid
    """
    pmesh = np.zeros(aprime_grid.shape)
    for i in range(len(aprime_grid)):
        for j in range(len(aprime_grid[0])):
            pmesh[i, j] = P[j, next_y_index]
    return pmesh

# def calc_sd_point(Pmesh, weight, sd_old) -> float:
#     """ 定常分布の一点を計算する関数

#     Args:
#         Pmesh (np.ndarray): 政策関数と遷移確率行列を使って計算した meshgrid
#         weight (np.ndarray): 前期の定常分布に係る重みづけ確率
#         sd_old (np.ndarray): 一期前の定常分布 f(a, y)

#     Returns:
#         float: 更新された(次期の)定常分布の一点の値
#     """
#     return np.sum(Pmesh * weight * sd_old)

@njit
def update_sd(aprime_grid: np.ndarray, agrid: np.ndarray, sd_old: np.ndarray, P: np.ndarray,
            split_prob_to_agrid = split_prob_to_agrid, calc_weight_matrix = calc_weight_matrix,
            gen_P = gen_pmesh,
            # calc_sd_point = calc_sd_point
            ) -> np.ndarray:
    """ 定常分布を更新する関数

    Args:
        aprime_grid (np.ndarray): 政策関数のグリッド
        agrid (np.ndarray): 資産のグリッド
        sd_old (np.ndarray): 更新前の定常分布
        P (np.ndarray): 遷移確率行列
        split_prob (Callable): 政策関数によって得られた a の値が、a_grid のどの点に移動するのかを示す確率ベクトルを計算する関数
        calc_weight (Callable): 前期の定常分布に係る重みづけ確率 weight を計算する関数
        gen_P (Callable): 遷移確率を 政策関数 aprime_grid との要素ごとの計算に使えるように meshgrid を生成する関数
        calc_sd_point (Callable): 定常分布の一点を計算する関数

    Returns:
        sd_new（np.ndarray）: 更新された定常分布のグリッド
    """
    sd_new = np.zeros(sd_old.shape)
    weight_matrix = calc_weight_matrix(aprime_grid, agrid, split_prob_to_agrid)
    for i in range(len(aprime_grid)):
        for j in range(len(aprime_grid[0])):
            p_mesh = gen_P(aprime_grid, P, j)
            # sd_new[i, j] = calc_sd_point(p_mesh, weight_matrix[:,:,i], sd_old)
            sd_new[i,j] = np.sum(p_mesh * weight_matrix[:,:,i] * sd_old)

    return sd_new

@njit
def sd_iteration(sd0: np.ndarray, aprime_grid: np.ndarray, agrid: np.ndarray, P: np.ndarray,
                        tol: float = 1e-6, max_iter: int = 1000,
                        update_sd = update_sd) -> np.ndarray:
    """ 定常分布を計算する関数

    Args:
        aprime_grid (np.ndarray): 政策関数のグリッド
        agrid (np.ndarray): 資産のグリッド
        guess_sd (np.ndarray): 初期の定常分布のグリッド
        P (np.ndarray): 遷移確率行列
        tol (float, optional): 許容する誤差. Defaults to 1e-6.
        max_iter (int, optional): 最大の反復回数. Defaults to 1000.
        update_sd (_type_, optional): 定常分布を更新する関数. Defaults to update_sd.

    Returns:
        np.ndarray: 収束した定常分布のグリッド
    """
    sd_old = sd0

    diff = tol + 1
    iteration = 0
    while diff > tol and iteration < max_iter:
        sd_new = update_sd(aprime_grid, agrid, sd_old, P)
        diff = np.max(np.abs(sd_new - sd_old))
        sd_old = sd_new
        iteration += 1
        # print(f"Iteration {iteration}: diff = {diff}")

    # 最終イタレーション回数の出力
    print(f"Total iterations until convergence: {iteration}")

    # 定常分布の出力
    # print("Stationary distribution f_{i,j}:")
    # print(sd_new)
    return sd_new
