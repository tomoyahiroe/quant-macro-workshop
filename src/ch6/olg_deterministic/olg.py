"""_summary_
search_equilibrium関数は、与えられた設定と初期金利を基に、経済モデルの均衡を探索します。
Args:
    st (Setting): モデルの設定を格納したSettingクラスのインスタンス。
    r0 (float): 初期金利。
    lambdaR (float): 金利調整のためのステップサイズ。
    DEBUG_MODE (bool, optional): デバッグモードを有効にするかどうか。デフォルトはFalse。
    ValueError: 金利が負の値になりモデルが破綻する場合に発生。
    Result: 均衡状態、収束パス、およびループ回数を格納したResultクラスのインスタンス。
この関数は以下の手順で均衡を探索します：
1. 初期金利を基に、労働需要、資本需要、賃金、生産量を計算します。
2. 保険料率と公的年金の支給額を計算します。
3. 個人の消費と資産の政策関数を計算します。
4. 資産の政策関数から総資本供給を計算します。
5. 資本需要と供給の差分が収束基準を満たすまでループを繰り返します。

Raises:
    ValueError: _description_

Returns:
    _type_: _description_
"""

from dataclasses import dataclass
import numpy as np
from setting import Setting

# 均衡に関する変数
@dataclass
class Equilibrium:
    """ Equilibrium
    """
    a_path: np.ndarray
    c_path: np.ndarray
    Kd: float
    Ks: float
    Ld: float
    Ls: float
    C: float
    Y: float
    r_star: float
    w_star: float
    K_star: float
    L_star: float
    tau: float # 年金保険料率
    p: float # 年金の給付額
    gc: float # 消費の成長率
    labor_market_error: float
    goods_market_error: float
    capital_market_error: float

# 結果を格納するクラス
@dataclass
class Result:
    """ Result of equilibrium
    """
    equilibrium: Equilibrium
    converge_path: np.ndarray
    loop: int

def search_equilibrium(st: Setting, Kd0: float, lambdaR: float, DEBUG_MODE = False) -> Result:
    """ Search equilibrium
    資本需要を与えられた初期値からスタートし、均衡を探索する関数。
    Args:
        st (Setting): モデルの設定を格納したSettingクラスのインスタンス。
        Kd0 (float): guessした資本需要。
        lambdaR (float): 金利調整のためのステップサイズ。
        DEBUG_MODE (bool, optional): デバッグモードを有効にするかどうか。デフォルトはFalse。
    """
    # 均衡クラスを初期化
    # このクラス内の変数をイテレーションの中で更新していく
    eq = Equilibrium(
        a_path = np.empty(st.J), 
        c_path = np.empty(st.J),
        Kd = Kd0,
        Ks = 0.0,
        Ld = 0.0,
        Ls = 0.0,
        C = 0.0,
        Y = 0.0,
        r_star = 0.0, # 金利の初期値
        w_star = 0.0,
        K_star = 0.0,
        L_star = 0.0,
        tau = 0.0,
        p = 0.0,
        gc = 0.0, # 消費の成長率
        labor_market_error = 0.0,
        goods_market_error = 0.0,
        capital_market_error = 0.0
    )

    converge_path = np.empty(0)


    diff = 1
    loop = 0
    while abs(diff) > st.tol:
        loop += 1

        # $K_d$ と企業の利潤最大化問題、個人の総労働供給 $L_s$から、価格 $w,r$を求める（と $Y$も求める）
        eq.Ls = float(np.sum(st.theta * st.mu))
        eq.Ld = eq.Ls
        eq.r_star = st.alpha * (eq.Kd / eq.Ld)**(st.alpha - 1) - st.delta
        eq.w_star = (1-st.alpha) * (eq.Kd**(st.alpha)) * (eq.Ld**(-st.alpha))
        print("w_star: ", eq.w_star, "r_star: ", eq.r_star)
        eq.Y = eq.Kd**st.alpha * eq.Ld**(1-st.alpha)
        print("Y: ", eq.Y, ", Ld: ", eq.Ld, ", Ls: ", eq.Ls)
        
        # 2. 保険料率 tau と公的年金の支給額 p を求める
        wbar = (eq.w_star * float(np.sum(st.mu * st.theta))) / (float(np.sum(st.mu)))
        eq.p = st.psi * wbar
        eq.tau = (st.psi*wbar*float(np.sum(st.mu[st.jr-1:]))) / (eq.w_star * (float(np.sum(st.mu[0:st.jr] * st.theta[0:st.jr]))))
        print("tau: ", eq.tau, ", p: ", eq.p)

        # 3. 個人の消費の成長率を求める
        eq.gc = (st.beta * (1 + eq.r_star))**(1/st.gamma) - 1
        print("gc: ", eq.gc)

        
        # 消費パスを計算
        # eq.c_path[0] = eq.Y / sum([((1-((1+eq.gc)/(1+eq.r_star))**j)/(1-(1+eq.gc)/(1+eq.r_star))) for j in range(st.J)])
        # 消費の現在価値の合計
        # c_pv = sum(((1 + eq.gc) / (1 + eq.r_star)) ** j for j in range(st.J))
        # c_pv = sum([((1-((1+eq.gc)/(1+eq.r_star))**j)/(1-(1+eq.gc)/(1+eq.r_star))) for j in range(st.J)]) # consumption present value
        # 生涯所得の現在価値
        # li_pv = sum([(st.theta[j]*eq.w_star)/(1+eq.r_star)**j for j in range(st.J)]) # lifetime income present value
        # 初期の初期の消費
        # eq.c_path[0] = li_pv/c_pv

        # 4. 生涯予算制約から消費を計算（修正版）
        # 生涯所得の現在価値
        lifetime_income = st.a1  # 初期資産
        for j in range(st.J):
            discount_factor = 1 / ((1 + eq.r_star) ** j)
            if j < st.jr:
                income = (1 - eq.tau) * st.theta[j] * eq.w_star
            else:
                income = eq.p
            lifetime_income += income * discount_factor

        # 消費の現在価値の合計
        consumption_pv_factor = sum(
            ((1 + eq.gc) / (1 + eq.r_star)) ** j for j in range(st.J)
        )
        
        # 初期消費
        eq.c_path[0] = lifetime_income / consumption_pv_factor
        
        for j in range(st.J):
            eq.c_path[j] = eq.c_path[0] * (1 + eq.gc) ** j

        eq.a_path[0] = st.a1
        
        for i in range(st.J-1):
            if i < st.jr:
                eq.a_path[i+1] = (1 + eq.r_star) * eq.a_path[i] + (1-eq.tau) * (st.theta[i] * eq.w_star) - eq.c_path[i]
            else:
                eq.a_path[i+1] = (1 + eq.r_star) * eq.a_path[i] + eq.p - eq.c_path[i]

        # 4. 資産の政策関数から総資本供給 $A$を計算する
        eq.Ks = float(np.sum(st.mu * eq.a_path))
        print("a_path: ", eq.a_path, ", c_path: ", eq.c_path)

        # 5. 所与の均衡金利から計算された資本と総資本供給の差分を取り、収束の基準より小さければ、均衡条件を満たしたとみなす
        diff = eq.Ks - eq.Kd
        print("diff: ", diff)
        converge_path = np.append(converge_path, diff)
        if DEBUG_MODE:
            print("loop: ", loop)
            print("r_star: ", eq.r_star, ", Ks: ", eq.Ks, ", diff: ", diff)
        
        eq.Kd = eq.Kd + lambdaR * diff
    
    # 元の資本需要に戻す
    eq.Kd = eq.Kd - lambdaR * diff
    eq.K_star = eq.Ks
    eq.L_star = eq.Ls
    eq.C = float(np.sum(st.mu * eq.c_path))
    return Result(eq, converge_path, loop)