import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.chebyshev import Chebyshev

# EGMを用いて TI を解くコード
# 教科書のグラフと比較するために p78 図3.6 のグラフと同様のパラメータを設定する
# u(c) = log(c), delta = 1.0, beta = 0.96, alpha = 0.4
# Aは確率的に変動する (A = 1.01, 0.99)

# STEP1&2&3&4: 
# - パラメータをカリブレーション
# - 収束の基準を定義
# - 技術水準の状態を定義
# - 確率遷移行列を手置き
class Parameters():
    '''
    モデルのパラメータをまとめたクラス
    '''
    def __init__(self,
                beta: float = 0.96,
                gamma: float = 1.0,
                alpha: float = 0.4,
                delta: float = 1.0,
                A: np.ndarray = np.array([1.01, 0.99]),
                P: np.ndarray = np.array([[0.875, 0.125], [0.125, 0.875]]),
                kp_max: float = 0.5,
                kp_min: float = 0.025,
                kp_size: int = 20,
                epsilon: float = 1e-5,
                max_iter: int = 10000):
        self.beta = beta
        self.gamma = gamma
        self.alpha = alpha
        self.delta = delta
        self.A = A
        self.P = P
        self.kp_max = kp_max
        self.kp_min = kp_min
        self.kp_size = kp_size
        self.epsilon = epsilon
        self.max_iter = max_iter

# 準備: 生産関数，富の関数，効用関数に関連する処理をまとめたクラスを定義
class ProductionFunc():
    '''
    生産関数を表すクラス
    '''
    def __init__(self, alpha: float = 0.4):
        self.alpha = alpha

    def output(self, k: float, A: float) -> float:
        '''
        生産関数の出力を計算する関数
        '''
        return A * (k ** self.alpha)

    def output_prime(self, k: float, A: float) -> float:
        '''
        生産関数の出力のkに関する一階導関数
        '''
        return self.alpha * A * (k ** (self.alpha - 1))

class WealthFunc():
    '''
    生産関数に減耗した資本を加えた，富の関数を表すクラス
    '''
    def __init__(self, prod_func: ProductionFunc, delta: float = 1.0):
        self.prod_func = prod_func
        self.delta = delta

    def wealth(self, k: float, A: float) -> float:
        '''
        富の関数
        '''
        return self.prod_func.output(k, A) + (1 - self.delta) * k

    def wealth_prime(self, k: float, A:float) -> float:
        '''
        富の関数のkに関する一階導関数
        '''
        return self.prod_func.output_prime(k, A) + (1 - self.delta)

class UtilityFunc():
    '''
    効用関数を表すクラス
    '''
    def __init__(self, gamma: float = 1.0):
        self.gamma = gamma

    def utility(self, c: float) -> float:
        '''
        効用関数
        '''
        if self.gamma == 1.0:
            return np.log(c)
        return (c ** (1 - self.gamma)) / (1 - self.gamma)

    def marginal_utility(self, c: float) -> float:
        '''
        効用関数の限界効用
        '''
        if self.gamma == 1.0:
            return 1 / c
        return c ** (-self.gamma)
    
    def inv_marginal_utility(self, mu: float) -> float:
        '''
        限界効用の逆関数
        '''
        if self.gamma == 1.0:
            return 1 / mu
        return mu ** (-1 / self.gamma)

# モデルを解くクラス，Model.solve()でモデルを解く
class Model():
    '''
    モデルを解くためのクラス    
    '''
    def __init__(self, params: Parameters):
        self.__params = params
        self.__prod_func = ProductionFunc(alpha=params.alpha)
        self.__wealth_func = WealthFunc(self.__prod_func, delta=params.delta)
        self.__utility_func = UtilityFunc(gamma=params.gamma)
        self.__old_pf_grid = np.zeros((len(self.__params.A), self.__params.kp_size))
        self.__new_pf_grid = np.zeros((len(self.__params.A), self.__params.kp_size))
        # STEP5: 制御変数のグリッドを定義
        self.__kp_grid = np.linspace(self.__params.kp_min, self.__params.kp_max, self.__params.kp_size)
        
        # モデルを解いた後に外部から参照したい変数を初期化
        self.policy_func = lambda m, A: 1.0 # 求まる予定の政策関数
        self.converged = False # 収束したかどうか

    def solve(self):
        '''
        モデルを解く
        '''
        # STEP6: 次期の資産 m' のグリッドを計算する
        mp = np.zeros((len(self.__params.A), self.__params.kp_size))
        for i, kp in enumerate(self.__kp_grid):
            for j, a in enumerate(self.__params.A):
                mp[i, j] = self.__wealth_func.wealth(kp, a)
        
        # STEP7: 初期値として政策関数を当て推量
