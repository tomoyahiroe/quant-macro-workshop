import numpy as np
from quantecon.optimize import brentq
from quantecon.markov import rouwenhorst
from interpolation import interp
from numba import njit, guvectorize
import matplotlib.pyplot as plt
import pandas as pd


# 指数グリッドをつくる関数
from numba import njit

@njit
def exponential_grid(a_min, a_max, N):
    """
    対数空間を使ったグリッド生成関数
    Parameters:
        a_min: float, グリッドの最小値
        a_max: float, グリッドの最大値
        N: int, グリッドの分割数
    Returns:
        grid: np.ndarray, 生成されたグリッド
    """
    # 対数変換
    log_a_min = np.log1p(a_min)  # log(1 + a_min)
    log_a_max = np.log1p(a_max)  # log(1 + a_max)

    # 等間隔のグリッドを生成
    log_grid = np.linspace(log_a_min, log_a_max, N)

    # 再変換して結果を格納
    grid = np.expm1(log_grid)  # exp(x) - 1
    return grid
    
# 線形補間を効率的に行うコード（interpolationパッケージのinterpでは正しい結果が得られなかった） 
# I took this function from state-space Jacobian package
@guvectorize(['void(float64[:], float64[:], float64[:], float64[:])'], '(n),(nq),(n)->(nq)')
def interpolate_y(x, xq, y, yq):
    """Efficient linear interpolation exploiting monotonicity.
    Complexity O(n+nq), so most efficient when x and xq have comparable number of points.
    Extrapolates linearly when xq out of domain of x.
    Parameters
    ----------
    x  : array (n), ascending data points
    xq : array (nq), ascending query points
    y  : array (n), data points
    Returns
    ----------
    yq : array (nq), interpolated points
    """
    nxq, nx = xq.shape[0], x.shape[0]

    xi = 0
    x_low = x[0]
    x_high = x[1]
    for xqi_cur in range(nxq):
        xq_cur = xq[xqi_cur]
        while xi < nx - 2:
            if x_high >= xq_cur:
                break
            xi += 1
            x_low = x_high
            x_high = x[xi + 1]

        xqpi_cur = (x_high - xq_cur) / (x_high - x_low)
        yq[xqi_cur] = xqpi_cur * y[xi] + (1 - xqpi_cur) * y[xi + 1]

# interpolationにinterpolate_yを使うバージョン
# 特定のアルゴリズムを実行して政策関数を更新する関数を出力する
# FOCを変更する場合やアルゴリズムを変更する場合はここを修正

# 今期の状態変数について繰り返し記号はi, 来期の状態変数について繰り返し記号はj

def EGMIteration(hp): # hpはSettingクラスからつくられるインスタンス

    # インスタンスからローカル変数を定義する
    R, beta, b, gamma, mutility = hp.R, hp.beta, hp.b, hp.gamma, hp.mutility
    a_grid, z_grid, Pz, w = hp.a_grid, hp.z_grid, hp.Pz, hp.w

    @njit
    def UpdatePF(h_old):

        h_new = np.empty_like(h_old)
        a_tilde = np.empty_like(a_grid)
        aprime_new = np.empty_like(a_grid)

        for i_z in range(len(z_grid)): # 今期の生産性について繰り返し
            z = z_grid[i_z]

            for j_a in range(len(a_grid)): # 来期の資産について繰り返し
                aprime = a_grid[j_a]

                expectation = 0
                for j_z in range(len(z_grid)): # 来期の生産性について繰り返し

                    # オイラー方程式の右辺を計算する
                    expectation += mutility(h_old[j_a, j_z]) * Pz[i_z, j_z]

                rhs = R * beta * expectation
                c_tilde = 1/((rhs)**(1/gamma))

                # 今期の資産を計算
                a_tilde[j_a] = (aprime - z*w + c_tilde)/R


            # 来期の資産の政策関数をinterpolationで求める
            interpolate_y(a_tilde, a_grid, a_grid, aprime_new)

            for i_a in range(len(a_grid)): # 今期の資産について繰り返し

                # 来期の資産の政策関数をinterpolationで求める
                #aprime_new = interp(a_tilde, a_grid, a_grid[i_a])

                # 借り入れ制約を考慮しつつ来期の資産を求める
                if aprime_new[i_a] < -b:
                    aprime_new[i_a] = -b
                else:
                    aprime_new[i_a] = aprime_new[i_a]

                # 消費の政策関数を求める
                h_new[i_a,i_z] = R * a_grid[i_a] + z*w - aprime_new[i_a]

        return h_new

    return UpdatePF
    


# メイン関数：特定のアルゴリズムでiterationを行い、問題を解く関数
# 基本的には変更する必要がない

def SolveProblem(hp,               # Settingクラスからつくられるインスタンス
                Algorithm,         # アルゴリズムを指定
                tol=1e-4,          # 許容繰り返し誤差
                max_iter=10000,     # iteration回数の最大値
                verbose=True,      # 進捗を表示するかどうか
                print_skip=25):    # 進捗を何回ごとに表示するか


    # インスタンスからローカル変数を定義する
    # R, beta, b, mutility = hp.R, hp.beta, hp.b, hp.mutility
    # a_grid, z_grid, Pz = hp.a_grid, hp.z_grid, hp.Pz
    lambdaPF = hp.lambdaPF
    hfun_old = hp.hfun_old

    # チェックのために外生変数のグリッドと遷移確率を表示する
    # print(f"About exogenous variables:")
    # print(f"grid is {z_grid}.")
    # print(f"Transition matrix is {Pz}.")



    # 政策関数を更新する関数を取得する
    UpdatePF = Algorithm(hp)

    # iterationを行い、問題を解く
    i = 0
    error = tol + 1

    while i < max_iter and error > tol:

        # 政策関数を更新する
        hfun_new_tilde = UpdatePF(hfun_old)

        # 古い政策関数と加重平均する
        hfun_new = lambdaPF*hfun_new_tilde + (1-lambdaPF)*hfun_old

        error = np.max(np.abs(hfun_new-hfun_old))
        i += 1
        if verbose and i % print_skip == 0:      # 進捗をprint_skip回ごとに表示する
            print(f"PF Error at iteration {i} is {error}.")
        hfun_old = hfun_new

    if i == max_iter:
        print("Failed to converge!")

    if verbose and i < max_iter:
        print(f"\nConverged in {i} iterations.")

    return hfun_new