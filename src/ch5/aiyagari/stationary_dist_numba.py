import numpy as np
from typing import Callable
from numba import njit

# TODO: 定常分布が政策関数と同じ要素数であることを前提としてしまっている

# 政策関数によって得られたグリッド外の値が、グリッド上に移動する確率行列を計算する関数
@njit
def split_prob_to_a_grid(avalue: float, agrid: np.ndarray) -> np.ndarray:
    """ 政策関数によって得られた a の値が、a_grid のどの点に移動するのかを示す確率ベクトルを計算する関数
    
    Args:
    a_value: 政策関数によって得られた a の値
    a_grid: a のグリッド
    return: a_value が a_grid のどの点に移動するのかを示す確率ベクトル 
    """
    probs = np.zeros(len(agrid))
    # a の値が a_grid の最小値より小さい場合
    if avalue <= agrid[0]:
        probs[0] = 1.0
        return probs

    # a の値が a_grid の最大値より大きい場合
    if avalue >= agrid[-1]:
        probs[-1] = 1.0
        return probs

    # a の値が a_grid の最小値より大きく最大値より小さい場合
    for i in range(len(agrid)-1):
        if agrid[i] <= avalue <= agrid[i+1]:
            probs[i] = (agrid[i+1] - avalue) / (agrid[i+1] - agrid[i])
            probs[i+1] = (avalue - agrid[i]) / (agrid[i+1] - agrid[i])
            return probs

    # この行までに return されていない場合はエラーを出す
    raise ValueError("Splitting probability failed.")

@njit
def calc_weight_grid(pfgrid: np.ndarray, agrid: np.ndarray, split_prob = split_prob_to_a_grid) -> np.ndarray:
    """ 前期の定常分布に係る重みづけ確率 weight を計算する関数

    Args:
        pfgrid: 政策関数のグリッド
        agrid: 資産のグリッド
    
    Returns:
        weight: 前期の定常分布に係る重みづけ確率のグリッド、shape は pfgrid と同じ
    """
    weight = np.empty((len(pfgrid), len(pfgrid[0]), len(agrid)))

    # pfgrid は 二次元配列
    for i in range(len(pfgrid)):
        for j in range(len(pfgrid[0])):
            weight[i, j] = split_prob(pfgrid[i, j], agrid)
    return weight

@njit
def gen_pmesh(pfgrid: np.ndarray, P: np.ndarray, next_y_index: int) -> np.ndarray:
    """ 遷移確率を 政策関数 pfgrid との要素ごとの計算に使えるように meshgrid を生成する関数

    Args:
        pfgrid (np.ndarray): 政策関数のグリッド
        P (np.ndarray): 遷移確率行列
        next_y_index (int): 次期の所得状態 y' のインデックス

    Returns:
        np.ndarray: 政策関数と遷移確率行列を使って計算するための meshgrid
    """
    pmesh = np.zeros(pfgrid.shape)
    for i in range(len(pfgrid)):
        for j in range(len(pfgrid[0])):
            pmesh[i, j] = P[j, next_y_index]
    return pmesh

@njit
def calc_sd_point(Pmesh, weight, sd_grid) -> float:
    """ 定常分布の一点を計算する関数

    Args:
        Pmesh (np.ndarray): 政策関数と遷移確率行列を使って計算した meshgrid
        weight (np.ndarray): 前期の定常分布に係る重みづけ確率のグリッド
        sd_grid (np.ndarray): 定常分布のグリッド f(a, y)

    Returns:
        float: 更新された(次期の)確率質量関数の一点の値
    """
    return np.sum(Pmesh * weight * sd_grid)

@njit
def update_sd(pfgrid: np.ndarray, agrid: np.ndarray, sd_grid: np.ndarray, P: np.ndarray,
            split_prob = split_prob_to_a_grid, calc_weight = calc_weight_grid,
            gen_P = gen_pmesh, calc_sdpoint = calc_sd_point) -> np.ndarray:
    """ 定常分布を更新する関数

    Args:
        pfgrid (np.ndarray): 政策関数のグリッド
        agrid (np.ndarray): 資産のグリッド
        sd_grid (np.ndarray): 更新前の定常分布のグリッド f(a, y)
        P (np.ndarray): 遷移確率行列
        split_prob (Callable): 政策関数によって得られた a の値が、a_grid のどの点に移動するのかを示す確率ベクトルを計算する関数
        calc_weight (Callable): 前期の定常分布に係る重みづけ確率 weight を計算する関数
        gen_P (Callable): 遷移確率を 政策関数 pfgrid との要素ごとの計算に使えるように meshgrid を生成する関数
        calc_sdpoint (Callable): 定常分布の一点を計算する関数

    Returns:
        np.ndarray: 更新された定常分布のグリッド
    """
    new_sd_grid = np.zeros(sd_grid.shape)
    weight = calc_weight(pfgrid, agrid, split_prob)
    for i in range(len(pfgrid)):
        for j in range(len(pfgrid[0])):
            p_mesh = gen_P(pfgrid, P, j)
            new_sd_grid[i, j] = calc_sdpoint(p_mesh, weight[:,:,i], sd_grid)
    return new_sd_grid

@njit
def solve_stationary_dist(pfgrid: np.ndarray, agrid: np.ndarray, guess_sd: np.ndarray, P: np.ndarray, 
                        tol: float = 1e-6, max_iter: int = 1000,
                        update_sd = update_sd) -> np.ndarray:
    """ 定常分布を計算する関数

    Args:
        pfgrid (np.ndarray): 政策関数のグリッド
        agrid (np.ndarray): 資産のグリッド
        guess_sd (np.ndarray): 初期の定常分布のグリッド
        P (np.ndarray): 遷移確率行列
        tol (float, optional): 許容する誤差. Defaults to 1e-6.
        max_iter (int, optional): 最大の反復回数. Defaults to 1000.
        update_sd (_type_, optional): 定常分布を更新する関数. Defaults to update_sd.

    Returns:
        np.ndarray: 収束した定常分布のグリッド
    """
    sd_grid = guess_sd
    
    diff = tol + 1
    iteration = 0
    while diff > tol and iteration < max_iter:
        new_sd_grid = update_sd(pfgrid, agrid, sd_grid, P)
        diff = np.max(np.abs(new_sd_grid - sd_grid))
        sd_grid = new_sd_grid
        iteration += 1
        print(f"Iteration {iteration}: diff = {diff}")
    
    # 最終イタレーション回数の出力
    print(f"Total iterations until convergence: {iteration}")

    # 定常分布の出力
    print("Stationary distribution f_{i,j}:")
    print(new_sd_grid)
    return new_sd_grid
