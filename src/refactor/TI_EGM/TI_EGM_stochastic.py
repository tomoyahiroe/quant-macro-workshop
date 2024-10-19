'''
EGMを用いて TI を解くコード
教科書のグラフと比較するために p78 図3.6 のグラフと同様のパラメータを設定する
u(c) = log(c), delta = 1.0, beta = 0.96, alpha = 0.4
Aは確率的に変動する (A = 1.01, 0.99)
'''
# TODO: クラス内のメソッドを static メソッドとして動作するように
# インスタンス変数を初期値として与えるように変更する

import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.chebyshev import Chebyshev


# STEP1&2&3&4:
# - 1. パラメータをカリブレーション
# - 2. 収束の基準 tol を定義
# - 3. 技術水準Aの状態を定義
# - 4. 確率遷移行列Pを手置き
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
    
    @staticmethod
    def Output_prime(k: float, A:float, alpha: float)-> float:
        return alpha * A * (k ** (alpha - 1))

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
        self.__kp_grid = np.linspace(self.__params.kp_min, 
                                    self.__params.kp_max,
                                    self.__params.kp_size)

        # モデルを解いた後に外部から参照したい変数を初期化
        self.policy_func = lambda m, A: 1.0 # 求まる予定の政策関数
        self.converged = False # 収束したかどうか
        self.loop = 0 # 収束までのループ回数

    def solve(self):
        '''
        モデルを解く
        '''
        if self.converged:
            print('Model has already converged.')
            return
        # STEP6: 次期の資産 m' のグリッドを計算する
        mp = np.zeros((self.__params.kp_size, len(self.__params.A)))
        for i, kp in enumerate(self.__kp_grid):
            for j, a in enumerate(self.__params.A):
                mp[i, j] = self.__wealth_func.wealth(kp, a)

        # STEP7: 初期値として政策関数を当て推量
        self.__old_pf_grid = np.copy(mp) * 0.5
        
        # STEP8: 次のステップを収束するまで繰り返す
        while self.loop < self.__params.max_iter and not self.converged:
            # オイラー方程式を解いて政策関数の更新
            self.__update_policy_func()
            np_policy_func = np.frompyfunc(self.policy_func, 2, 1)
            # 新しい政策関数（離散）を計算
            A_matrix = np.array(list(self.__params.A) * self.__params.kp_size).reshape(self.__params.kp_size, len(self.__params.A))
            self.__new_pf_grid = np_policy_func(mp, A_matrix).astype(np.float64)
            # 収束の確認
            self.__check_convergence()
            self.loop += 1

    def __interpolate(self, m_matrix,A_list,c_matrix):
        # TODO: チェビシェフ補間について、関数をより一般的なものに修正したい
        """チェビシェフ補間を行う関数を返す関数

        Args:
            m: 計算された m の行列 m[i, j]
            A (npt.NDArray): 説明変数 A の配列（離散化された状態の長さ分）
            c (npt.NDArray): 目的変数 c の配列
        """
        def interp_func(m,a):
            # A_list から aと一致する要素の番号を得る
            A_idx = np.where(A_list == a)[0][0]
            # m_matrixから k_matrix[i, z_idx]の配列を得る
            m_list = m_matrix[:, A_idx]
            c_list = c_matrix[:, A_idx]
            cheb_fit = Chebyshev.fit(m_list, c_list, deg=15)

            return cheb_fit(m)

        return interp_func

    def __update_policy_func(self):
        '''
        オイラー方程式を解いて政策関数を更新する関数
        '''
        # 可読性のために変数を再定義
        nk = self.__params.kp_size
        na = self.__params.A.size
        beta = self.__params.beta
        P = self.__params.P
        Kprime_grid = self.__kp_grid
        A_grid = self.__params.A
        u_prime = self.__utility_func.marginal_utility
        u_prime_inv = self.__utility_func.inv_marginal_utility
        f_tilde_prime = self.__wealth_func.wealth_prime
        Gamma = np.zeros((nk, na)) # オイラー方程式の右辺
        C_matrix = np.zeros((nk, na))
        M_matrix = np.zeros((nk, na))

        # オイラー方程式を解く
        for i in range(nk):
            for j in range(na):
                for k in range(na):
                    Gamma[i, j] += beta * P[j, k] * u_prime(self.__old_pf_grid[i, k]) * f_tilde_prime(Kprime_grid[i], A_grid[k])
                C_matrix[i, j] = u_prime_inv(Gamma[i, j])
                M_matrix[i, j] = C_matrix[i,j] + Kprime_grid[i]

        # 補間をして政策関数を更新
        self.policy_func = self.__interpolate(M_matrix,A_grid,C_matrix)
        return

    def __check_convergence(self):
        '''
        収束判定
        '''
        # np.allclose: 2つの配列が等しいかどうかを判定する関数
        if np.allclose(self.__old_pf_grid, self.__new_pf_grid, atol=self.__params.epsilon):
            self.converged = True
        else:
            self.__old_pf_grid = np.copy(self.__new_pf_grid)