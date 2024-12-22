import numpy as np
import quantecon
import interpolation
from numba import njit
import setting as st


# 特定のアルゴリズムを実行して政策関数を更新する関数を出力する
# FOCを変更する場合やアルゴリズムを変更する場合はここを修正

# 今期の状態変数について繰り返し記号はi, 来期の状態変数について繰り返し記号はj

def TimeIteration(hp: st.Setting): # hpはSettingクラスからつくられるインスタンス

    # インスタンスからローカル変数を定義する
    R, beta, b, mutility, w = hp.R, hp.beta, hp.b, hp.mutility, hp.w
    a_grid, z_grid, Pz = hp.a_grid, hp.z_grid, hp.Pz

    @njit
    def FOCs(c, a, z, i_z, hfun):

        # 制約式から次期の内生的な状態変数を計算する
        aprime = R * a + w*z - c

        expectation = 0
        for j_z in range(len(z_grid)):
            # 政策関数の候補を補間して次期の制御変数を計算する
            cprime = interpolation.interp(a_grid, hfun[:, j_z], aprime)
            if cprime is None:
                raise ValueError("interp returned None, but a float is expected.")

            # オイラー方程式の右辺を計算する
            expectation += mutility(cprime) * Pz[i_z, j_z]

        rhs = max(R * beta * expectation, mutility(R * a + w*z + b))

        FOC_diff = mutility(c) - rhs

        return FOC_diff


    @njit
    def UpdatePF(h_old):

        h_new = np.empty_like(h_old)
        for i_a in range(len(a_grid)):
            a = a_grid[i_a]
            for i_z in range(len(z_grid)):
                z = z_grid[i_z]
                c_star = quantecon.optimize.root_finding.brentq(FOCs, 1e-8, R * a + w*z + b, args=(a, z, i_z, h_old)).root
                h_new[i_a, i_z] = c_star

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
            print(f"Error at iteration {i} is {error}.")
        hfun_old = hfun_new

    if i == max_iter:
        print("Failed to converge!")

    if verbose and i < max_iter:
        print(f"\nConverged in {i} iterations.")

    return hfun_new