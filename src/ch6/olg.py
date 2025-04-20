from dataclasses import dataclass
from operator import eq
import numpy as np
from setting import Setting
from utils import *

# 均衡に関する変数
@dataclass
class Equilibrium:
    """ Equilibrium
    """
    pf_aprime: np.ndarray
    pf_c: np.ndarray
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

# 結果を格納するクラス
@dataclass
class Result:
    """ Result of equilibrium
    """
    eq: Equilibrium
    converge_path: np.ndarray
    loop: int

def search_equilibrium(st: Setting, r0: float, lambdaR: float, DEBUG_MODE = False) -> Result:
    """ Search equilibrium

    """
    # 均衡クラスを初期化
    # このクラス内の変数をイテレーションの中で更新していく
    eq = Equilibrium(
        pf_aprime = np.empty(st.J),
        pf_c = np.empty(st.J),
        Kd = 0.0,
        Ks = 0.0,
        Ld = 0.0,
        Ls = 0.0,
        C = 0.0,
        Y = 0.0,
        r_star = r0, # 金利の初期値
        w_star = 0.0,
        K_star = 0.0,
        L_star = 0.0,
        tau = 0.0,
        p = 0.0,
        gc = 0.0, # 消費の成長率
    )

    converge_path = np.empty(0)


    diff = 1
    loop = 0
    while abs(diff) > st.tol:
        if 1 + eq.r_star <= 0:
            raise ValueError("r_star too small, model broken")
        loop += 1

        # 1. 均衡金利 r_star, 企業の利潤最大化条件, 労働供給を所与として、
        # 　　労働需要 Ld、総資本需要 Kd, 賃金 w, 生産量 Y を求める
        eq.Ls = float(np.sum(st.theta * st.mu))
        eq.Ld = eq.Ls
        eq.Kd = ((eq.r_star + st.delta) / (st.alpha * (eq.Ld**(1-st.alpha)))) ** (1 / (st.alpha - 1))
        print("Kd: ", eq.Kd, ", Ks: ", eq.Ks)
        eq.w_star = st.alpha * (eq.Kd**(st.alpha)) * (eq.Ld**(-st.alpha))
        print("w_star: ", eq.w_star, "r_star: ", eq.r_star)
        eq.Y = eq.Kd**st.alpha * eq.Ld**(1-st.alpha)
        print("Y: ", eq.Y, ", Ld: ", eq.Ld, ", Ls: ", eq.Ls)
        
        # 2. 保険料率 tau と公的年金の支給額 p を求める
        wbar = (eq.w_star * float(np.sum(st.mu * st.theta))) / (float(np.sum(st.mu)))
        eq.p = st.psi * wbar
        eq.tau = (st.psi*wbar*float(np.sum(st.mu[st.jr-1:]))) / (eq.w_star * (float(np.sum(st.mu[0:st.jr] * st.theta[0:st.jr]))))
        print("tau: ", eq.tau, ", p: ", eq.p)

        # 3. 個人の消費と資産の政策関数を求める
        eq.gc = (st.beta * (1 + eq.r_star))**(1/st.gamma) -1
        print("gc: ", eq.gc)

        
        # 消費の政策関数を計算
        eq.pf_c[0] = eq.Y /((1-((1+eq.gc)/(1+eq.r_star))**st.J)/(1-(1+eq.gc)/(1+eq.r_star)))
        eq.pf_c[:] = [eq.pf_c[0] * (1+eq.gc)**i for i in range(st.J)]
        eq.pf_aprime[0] = st.a1
        print("test", (1 + eq.r_star) * eq.pf_aprime[0] + (1-eq.tau) * (st.theta[0] * eq.w_star) - eq.pf_c[0])
        for i in range(st.J-1):
            if i < st.jr:
                eq.pf_aprime[i+1] = (1 + eq.r_star) * eq.pf_aprime[i] + (1-eq.tau) * (st.theta[i] * eq.w_star) - eq.pf_c[i]
            else:
                eq.pf_aprime[i+1] = (1 + eq.r_star) * eq.pf_aprime[i] + eq.p - eq.pf_c[i]

        # 4. 資産の政策関数から総資本供給 $A$を計算する
        eq.Ks = float(np.sum(st.mu * eq.pf_aprime))
        print("pf_aprime: ", eq.pf_aprime, ", pf_c: ", eq.pf_c)

        # 5. 所与の均衡金利から計算された資本と総資本供給の差分を取り、収束の基準より小さければ、均衡条件を満たしたとみなす
        diff = eq.Ks - eq.Kd
        print("diff: ", diff)
        converge_path = np.append(converge_path, diff)
        if DEBUG_MODE:
            print("loop: ", loop)
            print("r_star: ", eq.r_star, ", Ks: ", eq.Ks, ", diff: ", diff)
        
        eq.r_star = eq.r_star - lambdaR * diff
    
    eq.r_star = eq.r_star + lambdaR * diff
    eq.K_star = eq.Ks
    eq.L_star = eq.Ls
    eq.C = float(np.sum(st.mu * eq.pf_c))
    return Result(eq, converge_path, loop)