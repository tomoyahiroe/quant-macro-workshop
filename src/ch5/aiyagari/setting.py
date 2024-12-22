from typing import Callable
import quantecon
import numpy as np
from numba import njit
import scipy

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
                w=0,                         # 賃金の初期化
                r0 = 0.03,                      # 利子率の初期値
                alpha = 0.36,                    # 資本分配率
                delta = 0.05,                    # 固定資本減耗率
                lambdaR = 0.001,                 # 利子率の更新度
                lambdaPF = 1):                 # 政策関数の更新度

        # パラメータを設定する
        self.R = R
        self.beta = beta
        self.b = b
        self.gamma = gamma
        self.alpha = alpha
        self.delta = delta
        self.sigma = sigma
        self.rho = rho
        self.r0 = r0

        # 外生変数の遷移確率とグリッドを設定する
        mc = quantecon.markov.approximation.rouwenhorst(nz, rho, sigma, mu)
        self.Pz = np.array(mc.P)
        if mc.state_values is None:
            raise ValueError("mc.state_values is None, cannot apply np.exp.")
        self.z_grid = np.exp(mc.state_values)

        # 内生的な状態変数のグリッドを設定する
        a_grid = np.linspace(-b, a_max, na)
        self.a_grid = a_grid

        # 賃金を設定
        self.w = w

        # 利子率の更新度を設定する
        self.lambdaR = lambdaR

        # 政策関数の更新度を設定する
        self.lambdaPF = lambdaPF

        # CRRA型効用関数と限界効用を定義する
        gamma = self.gamma
        if gamma == 1:
            self.utility = np.log
            self.mutility: Callable[[float], float] = njit(lambda x: 1 / x)
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