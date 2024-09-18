import abc
from email import policy
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import leastsq


class AbstractModel(metaclass=abc.ABCMeta):
    """Modelクラスの呼び出せる関数を定義するための抽象クラス"""
    @abc.abstractmethod
    def get_g2_theta(self) -> npt.NDArray: # 第2期の政策関数を決定するパラメータ theta を返す関数
        pass
    # @abc.abstractmethod
    # def get_g1_eta(self) -> npt.NDArray: # 第1期の政策関数を決定するパラメータ eta を返す関数
    #     pass
    @abc.abstractmethod
    def policy_func_g2(self, a_2: float) -> float: # a_3 = g2(a_2)を実装した関数
        pass
    # @abc.abstractmethod
    # def policy_func_g1(self, a_1: float) -> float: # a_2 = g1(a_1)を実装した関数
    #     pass


class Model(AbstractModel):
    """
    3期間のモデルを解くために必要なパラメータと関数をまとめたクラス
    """
    def __init__(self,
        beta = 0.985 ** 20, # 割引因子
        gamma = 2.0, # 相対的リスク回避度
        r = 1.025 ** 20 - 1, # 利子率
        y_1 = 1.0, # 外生的に与えられる若年期の所得
        y_2 = 1.2, # 外生的に与えられる中年期の所得
        y_3 = 0.4, # 外生的に与えられる老年期の年金
        a_min = 0.025, # 貯蓄の評価点の下限
        a_max = 2.0, # 貯蓄の評価点の上限
        a_size = 80, # 貯蓄の評価点の個数
        initial_guess = [0.1, 0.1] # 政策関数を最適化する際の初期値
        ):

        self.beta, self.gamma, self.r = beta, gamma, r
        self.y_1, self.y_2, self.y_3 = y_1, y_2, y_3
        self.a_min, self.a_max, self.a_size = a_min, a_max, a_size
        self.initial_guess = initial_guess
        self.a_grid = np.linspace(a_min, a_max, a_size, dtype=np.float64)

        self.__theta0, self.__theta1 = [None, None] # 第2期の政策関数を決定するパラメータ
        self.__eta0, self.__eta1 = [None, None] # 第1期の政策関数を決定するパラメータ
    

    def __mu_CRRA(self, c: float) -> float:
        """CRRA型の効用関数の限界効用
        Args:
            c (float): 消費量
        Returns:
            float: 限界効用水準
        """
        # if c < 0:
        #     mu = - 10 ** 5 + c
        mu = c ** (-self.gamma)
        return mu


    def __residual(self, c_early: float, c_late: float) -> float:
        """オイラー方程式の残差
        Args:
            c_early (float): 前期の消費量
            c_late (float): 後期の消費量
        Returns:
            float: オイラー方程式の残差
        """
        residual = self.beta * (1 + self.r) * self.__mu_CRRA(c_late) / self.__mu_CRRA(c_early) - 1
        return residual


    def __calc_c3(self, a_3: float) -> float:
        """第3期の消費量を計算する関数
        Args:
            a_3 (float): 第2期に貯蓄して第3期に持ち越された資産
        Returns:
            float: 第3期の消費量
        """
        c_3 = self.y_3 + (1 + self.r) * a_3
        return c_3


    def __calc_c2(self, a_2: float, a_3: float) -> float:
        """第2期の消費量を計算する関数
        Args:
            a_2 (float): 第1期に貯蓄して第2期に持ち越された資産
            a_3 (float): 第2期に貯蓄して第3期に持ち越された資産
        Returns:
            float: 第2期の消費量
        """
        c_2 = self.y_2 + (1 + self.r) * a_2 - a_3
        return c_2
    
    
    def __calc_c1(self, a_1: float, a_2: float) -> float:
        """第1期の消費量を計算する関数
        Args:
            a_1 (float): 第0期に貯蓄して第1期に持ち越された資産
            a_2 (float): 第1期に貯蓄して第2期に持ち越された資産
        Returns:
            float: 第1期の消費量
        """
        c_1 = self.y_1 + a_1 - a_2
        return c_1


    def __calc_a3(self, theta: npt.NDArray, a_2: float) -> float:
        """第3期に繰り越される資産を計算する関数
        Descriptions:
            政策関数ではない. 引数に渡された係数の配列と第2期の資産の値から第3期の資産を計算する.
        Args:
            theta (npt.NDArray): [theta0, theta1]に分割代入される係数の配列
            a_2 (float): 第2期の資産
        Returns:
            float: 第3期の資産
        """
        theta0, theta1 = theta
        a_3 = theta0 + theta1 * a_2
        return a_3

    def __calc_a2(self, eta: npt.NDArray, a_1: float) -> float:
        """第2期に繰り越される資産を計算する関数
        Descriptions:
            政策関数ではない. 引数に渡された係数の配列と第1期の資産の値から第2期の資産を計算する.
        Args:
            eta (npt.NDArray): [eta0, eta1]に分割代入される係数の配列
            a_1 (float): 第1期の資産
        Returns:
            float: 第2期の資産
        """
        eta0, eta1 = eta
        a_2 = eta0 + eta1 * a_1
        return a_2


    def __calc_residuals_g2(self, theta: npt.NDArray) -> npt.NDArray:
        """評価点におけるオイラー方程式の残差を配列として返す関数
        Args:
            param (np.array): 政策関数の係数を表す配列
        Returns:
            np.array: 評価点におけるオイラー方程式の残差
        """
        residuals = np.zeros(self.a_size)
        for i, a_2 in enumerate(self.a_grid):
            a_3 = self.__calc_a3(theta, a_2)
            if a_3 < 0: # 資産が負になる場合は, ペナルティを与える
                residuals[i] = 10 ** 5 - a_3
                continue
            c_3 = self.__calc_c3(a_3)
            if c_3 < 0: # 消費が負になる場合は, ペナルティを与える
                residuals[i] = 10 ** 5 - c_3
                continue
            c_2 = self.__calc_c2(a_2, a_3)
            if c_2 < 0: # 消費が負になる場合は, ペナルティを与える
                residuals[i] = 10 ** 5 - c_2
                continue
            residuals[i] = self.__residual(c_2, c_3)
        
        return residuals
    
    
    def __calc_residuals_g1(self, eta: npt.NDArray) -> npt.NDArray:
        """評価点におけるオイラー方程式の残差を配列として返す関数
        Args:
            param (np.array): 政策関数の係数を表す配列
        Returns:
            np.array: 評価点におけるオイラー方程式の残差
        """
        residuals = np.zeros(self.a_size)
        for i, a_1 in enumerate(self.a_grid):
            a_2 = self.__calc_a2(eta, a_1)
            if a_2 < 0:
                residuals[i] = 10 ** 5 - a_2
                continue
            c_2 = self.__calc_c2(a_2, self.policy_func_g2(a_2))
            if c_2 < 0:
                residuals[i] = 10 ** 5 - c_2
                continue
            c_1 = self.__calc_c1(a_1, a_2)
            if c_1 < 0:
                residuals[i] = 10 ** 5 - c_1
                continue
            residuals[i] = self.__residual(c_1, c_2)
        return residuals


    def get_g2_theta(self) -> npt.NDArray:
        """第2期の政策関数を決定するパラメータ theta を返す関数
        Returns:
            npt.NDArray: [theta0, theta1]の係数を返す
        """
        if (self.__theta0 is not None) and (self.__theta1 is not None):
            return np.array([self.__theta0, self.__theta1])
        result = leastsq(self.__calc_residuals_g2, x0 = self.initial_guess)
        self.__theta0, self.__theta1 = result[0]
        return np.array([self.__theta0, self.__theta1])


    def get_g1_eta(self) -> npt.NDArray:
        """第1期の政策関数を決定するパラメータ eta を返す関数
        Returns:
            npt.NDArray: [eta0, eta1]の係数を返す
        """
        if (self.__eta0 is not None) and (self.__eta1 is not None):
            return np.array([self.__eta0, self.__eta1])
        result = leastsq(self.__calc_residuals_g1, x0 = self.initial_guess)
        self.__eta0, self.__eta1 = result[0]
        return np.array([self.__eta0, self.__eta1])


    def policy_func_g2(self, a_2: float) -> float:
        """第2期の政策関数(貯蓄関数), 第2期の資産から第3期の最適な資産を計算する
        Description:
            - self.__theta0, self.__theta1 の値が None の場合アルゴリズムを実行する
            - self.__theta0, self.__theta1 に値がある場合は, これを用いてa_3を計算する
        Args:
            a_2 (float): 第2期の資産
        Returns:
            a_3 (float): 第3期の資産
        """
        if (self.__theta0 is not None) and (self.__theta1 is not None):
            # すでに政策関数が求まっている場合は, そのまま a_3 を計算する
            a_3 = self.__theta0 + self.__theta1 * a_2
        else:
            # 政策関数が求まっていない場合は, 最適化を行う
            theta0, theta1 = self.get_g2_theta()
            a_3 = theta0 + theta1 * a_2
        
        return a_3
    
    
    def policy_func_g1(self, a_1: float) -> float:
        """第1期の政策関数(貯蓄関数), 第1期の資産から第2期の最適な資産を計算する
        Description:
            - self.__eta0, self.__eta1 の値が None の場合アルゴリズムを実行する
            - self.__eta0, self.__eta1 に値がある場合は, これを用いてa_2を計算する
        Args:
            a_1 (float): 第1期の資産
        Returns:
            a_2 (float): 第2期の資産
        """
        if (self.__eta0 is not None) and (self.__eta1 is not None):
            # すでに政策関数が求まっている場合は, そのまま a_2 を計算する
            a_2 = self.__eta0 + self.__eta1 * a_1
        else:
            # 政策関数が求まっていない場合は, 最適化を行う
            eta0, eta1 = self.get_g1_eta()
            a_2 = eta0 + eta1 * a_1
        
        return a_2


m = Model()
theta0, theta1 = m.get_g2_theta()
eta0, eta1 = m.get_g1_eta()
print(f"a_2 = g1(a_1) = {eta0} + {eta1} * a_1")
print(f"a_3 = g2(a_2) = {theta0} + {theta1} * a_2")

# 制作関数を二つ並べてプロット
fig, ax = plt.subplots()
ax.plot(m.a_grid, [m.policy_func_g1(a) for a in m.a_grid], label="Policy function g1")
ax.plot(m.a_grid, [m.policy_func_g2(a) for a in m.a_grid], label="Policy function g2")
ax.legend()
plt.show()
