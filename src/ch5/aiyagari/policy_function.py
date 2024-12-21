import numpy as np
from quantecon.optimize import brentq
from quantecon.markov import rouwenhorst
from interpolation import interp
from numba import njit
import pandas as pd


# 解く問題の設定を行う（パラメータ, グリッド, 効用関数や生産関数）
# モデルを変更する場合にはここを修正
class Setting:

    def __init__(self,
                 R=1.01,                          # 粗実質利子率
                 beta=0.99,                          # 割引因子
                 gamma=1,                          # 相対的リスク回避度(異時点間の代替弾力性の逆数)
                 b=0,                             # 内生的な状態変数の最小値, 借入制約
                 a_max=16,                        # 内生的な状態変数の最大値
                 na=21,                           # 内生的な状態変数のグリッド数
                 mu=0,                            # 外生変数のAR(1)過程の定数項
                 rho=0.6,                        # 外生変数のAR(1)過程の慣性
                 sigma=0.4,                       # 外生変数のAR(1)過程のショック項の標準偏差
                 nz = 11,                         # 外生変数のグリッド数
                 lambdaPF = 1):                 # 政策関数の更新度

        # パラメータを設定する
        self.R = R
        self.beta = beta
        self.b = b
        self.gamma = gamma

        # 外生変数の遷移確率とグリッドを設定する
        mc = rouwenhorst(nz, rho, sigma, mu)
        self.Pz = mc.P
        self.z_grid = np.exp(mc.state_values)

        # 内生的な状態変数のグリッドを設定する
        a_grid = np.linspace(-b, a_max, na)
        self.a_grid = a_grid

        # 政策関数の更新度を設定する
        self.lambdaPF = lambdaPF

        # CRRA型効用関数と限界効用を定義する
        gamma = self.gamma
        if gamma == 1:
            self.utility = np.log
            self.mutility = njit(lambda x: 1 / x)
        else:
            self.utility = njit(lambda x: x**(1-gamma) / (1 - gamma))
            self.mutility = njit(lambda x: x**(-gamma))


		    # 政策関数の初期値を定義する
        z_grid = self.z_grid
        hfun_old = np.empty((len(a_grid), len(z_grid)))
        for i_a, a in enumerate(a_grid):
            for i_z, z in enumerate(z_grid):
                c_max = 0.5* (R * a + z + b)
                hfun_old[i_a, i_z] = c_max
        self.hfun_old = hfun_old


# 特定のアルゴリズムを実行して政策関数を更新する関数を出力する
# FOCを変更する場合やアルゴリズムを変更する場合はここを修正

# 今期の状態変数について繰り返し記号はi, 来期の状態変数について繰り返し記号はj

def TimeIteration(hp): # hpはSettingクラスからつくられるインスタンス

    # インスタンスからローカル変数を定義する
    R, beta, b, mutility = hp.R, hp.beta, hp.b, hp.mutility
    a_grid, z_grid, Pz = hp.a_grid, hp.z_grid, hp.Pz

    @njit
    def FOCs(c, a, z, i_z, hfun):

        # 制約式から次期の内生的な状態変数を計算する
        aprime = R * a + z - c

        expectation = 0
        for j_z in range(len(z_grid)):
            # 政策関数の候補を補間して次期の制御変数を計算する
            cprime = interp(a_grid, hfun[:, j_z], aprime)

            # オイラー方程式の右辺を計算する
            expectation += mutility(cprime) * Pz[i_z, j_z]

        rhs = max(R * beta * expectation, mutility(R * a + z + b))

        FOC_diff = mutility(c) - rhs

        return FOC_diff


    @njit
    def UpdatePF(h_old):

        h_new = np.empty_like(h_old)
        for i_a in range(len(a_grid)):
            a = a_grid[i_a]
            for i_z in range(len(z_grid)):
                z = z_grid[i_z]
                c_star = brentq(FOCs, 1e-8, R * a + z + b, args=(a, z, i_z, h_old)).root
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
    R, beta, b, mutility = hp.R, hp.beta, hp.b, hp.mutility
    a_grid, z_grid, Pz = hp.a_grid, hp.z_grid, hp.Pz
    lambdaPF = hp.lambdaPF
    hfun_old = hp.hfun_old

    # チェックのために外生変数のグリッドと遷移確率を表示する
    print(f"About exogenous variables:")
    print(f"grid is {z_grid}.")
    print(f"Transition matrix is {Pz}.")



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


hp = Setting(beta=0.97, gamma = 2, b = 3, lambdaPF = 1, na = 100, nz = 2, rho = 0.53, sigma = 0.2510, R = 1.0276)
hfun_c = SolveProblem(hp,TimeIteration)

