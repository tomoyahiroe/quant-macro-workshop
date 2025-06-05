from tabnanny import verbose
from typing import Callable
import quantecon
from quantecon.markov import tauchen
import numpy as np
from numba import njit

# モデルの設定を定義するクラス
# イテレーションを通して値が変わるような変数はここに定義しない
# モデルを変更する場合にはここを修正
class Setting:

    def __init__(self,
                beta=0.98,                       # 割引因子
                gamma=1,                         # 相対的リスク回避度(異時点間の代替弾力性の逆数)
                # b=3,                             # 内生的な状態変数の最小値, 借入制約
                # a_max=16,                        # 内生的な状態変数の最大値
                # na=21,                           # 内生的な状態変数のグリッド数
                alpha = 0.4,                    # 資本分配率
                delta = 0.08,                    # 固定資本減耗率
                psi = 0.5,                       # 年金の平均所得代替率
                J = 65,                          # モデルの期間
                jw = 20,                         # 勤労期の初期 j work
                jr = 46,                         # 引退期の初期 j retire
                a1 = 0.0,                        # 初期資産
                tol = 1e-5,                     # 収束判定の閾値
                ):                   

        # パラメータを設定する
        self.beta = beta
        # self.b = b
        self.gamma = gamma
        self.alpha = alpha
        self.delta = delta
        # self.na = na
        # self.a_min = -b
        # self.a_max = a_max
        self.psi = psi
        self.J = J
        self.jw = jw
        self.jr = jr
        self.a1 = a1
        self.tol = tol

        # 内生的な状態変数のグリッドを設定する
        # a_grid = np.linspace(-b, a_max, na)
        # a_grid = maliar_grid(-b, a_max, na, theta = 2.0)
        # self.a_grid = a_grid

        # CRRA型効用関数と限界効用を定義する
        gamma = self.gamma
        if gamma == 1:
            self.utility = np.log
            self.mutility: Callable[[float], float] = njit(lambda x: 1 / x)
        else:
            self.utility = njit(lambda x: x**(1-gamma) / (1 - gamma))
            self.mutility = njit(lambda x: x**(-gamma))

        # 人口の分布 65要素全て１
        self.mu = np.ones(J)

        # 各年齢の労働生産性 65要素全て1
        self.theta = np.ones(J)