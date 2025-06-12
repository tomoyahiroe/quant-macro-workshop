"""
Settingクラスは、オーバーラッピング・ジェネレーションズ（OLG）モデルの基本的なパラメータ設定を管理するクラスです。

主な機能:
- 割引因子、リスク回避度、資本分配率、減耗率、年金の所得代替率など、モデルの主要パラメータを初期化します。
- モデルの期間や労働・引退開始年齢、初期資産、収束判定の閾値も設定できます。
- 効用関数（CRRA型）と限界効用関数をパラメータに応じて定義します。
- 人口分布（mu）と各年齢の労働生産性（theta）を全て1で初期化します。

注意:
- モデルのイテレーションを通じて変化する変数はこのクラスには含めません。
- モデルの構造を変更する場合は、このクラスのパラメータを修正してください。
"""

from typing import Callable
import numpy as np
from numba import njit


class Setting:
    """
    設定クラス（Setting）
    このクラスは、OLG（Overlapping Generations）モデルにおける主要なパラメータを管理します。
    割引因子、リスク回避度、資本分配率、減耗率、年金の所得代替率、モデル期間、労働・引退開始年齢、初期資産、収束判定閾値などを属性として保持します。
    また、CRRA型効用関数および限界効用関数を定義し、人口分布および各年齢の労働生産性を初期化します。
    Attributes:
        beta (float): 割引因子
        gamma (float): 相対的リスク回避度（異時点間の代替弾力性の逆数）
        alpha (float): 資本分配率
        delta (float): 固定資本減耗率
        psi (float): 年金の平均所得代替率
        J (int): モデルの期間（世代数）
        jw (int): 勤労期の初期年齢
        jr (int): 引退期の初期年齢
        a1 (float): 初期資産
        tol (float): 収束判定の閾値
        utility (callable): CRRA型効用関数
        mutility (callable): 限界効用関数
        mu (np.ndarray): 各年齢の人口分布
        theta (np.ndarray): 各年齢の労働生産性
    """

    def __init__(
        self,
        beta=0.98,  # 割引因子
        gamma=1,  # 相対的リスク回避度(異時点間の代替弾力性の逆数)
        alpha=0.4,  # 資本分配率
        delta=0.08,  # 固定資本減耗率
        psi=0.5,  # 年金の平均所得代替率
        J=61,  # モデルの期間
        jw=20,  # 勤労期の初期 j work
        jr=46,  # 引退期の初期 j retire
        a1=0.0,  # 初期資産
        tol=1e-5,  # 収束判定の閾値
    ):

        # パラメータを設定する
        self.beta = beta
        self.gamma = gamma
        self.alpha = alpha
        self.delta = delta
        self.psi = psi
        self.J = J
        self.jw = jw
        self.jr = jr
        self.a1 = a1
        self.tol = tol

        # CRRA型効用関数と限界効用を定義する
        gamma = self.gamma
        if gamma == 1:
            self.utility = np.log
            self.mutility: Callable[[float], float] = njit(lambda x: 1 / x)
        else:
            self.utility = njit(lambda x: x ** (1 - gamma) / (1 - gamma))
            self.mutility = njit(lambda x: x ** (-gamma))

        # 人口の分布 65要素全て１
        self.mu = np.ones(J)

        # 各年齢の労働生産性 65要素全て1
        self.theta = np.ones(J)
