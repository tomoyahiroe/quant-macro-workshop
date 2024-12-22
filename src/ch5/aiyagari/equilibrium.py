from dataclasses import dataclass
import numpy as np
from policy_function import SolveProblem, TimeIteration
from stationary_dist import sd_iteration
from setting import Setting

@dataclass
class Result:
    """ Result of equilibrium
    """
    r_star: float
    w_star: float
    K_star: float
    hfun_c: np.ndarray
    hfun_a: np.ndarray
    sd: np.ndarray
    converge_path: np.ndarray
    loop: int

def search_equilibrium(hp: Setting, DEBUG_MODE = False) -> Result:
    """ Search equilibrium
    """
    # ローカル変数を定義
    alpha, beta, gamma = hp.alpha, hp.beta, hp.gamma, 
    delta, b, rho, sigma = hp.delta, hp.b, hp.rho, hp.sigma
    r0 = hp.r0
    lambdaR = hp.lambdaR


    converge_path = np.empty(0)

    diff = 1
    loop = 0
    while abs(diff) > 1e-6:
        loop += 1
        # 1. 企業の利潤最大化条件から 総資本需要 K0d, 賃金 wage を求める
        Kd = ((r0 + delta) / alpha) ** (1 / (alpha - 1))
        wage = (1 - alpha) * (Kd ** alpha)

        # 2. 個人の最適化問題を解いて 政策関数を求める
        hp = Setting(beta=beta, gamma = gamma, b = b, lambdaPF = 1, na = 20, nz = 3, rho = rho, sigma = sigma, R = 1 + r0, w = wage)
        hfun_c = SolveProblem(hp,TimeIteration)

        # 3. 定常分布を求める
        # hfun_c から 時期のアセット aの政策関数を求める
        hfun_aprime = np.empty((len(hp.a_grid), len(hp.z_grid)))
        a_mesh, z_mesh = np.meshgrid(hp.a_grid, hp.z_grid, indexing='ij') # ユニバーサル関数を使用するためのグリッドを生成
        hfun_aprime = hp.R * a_mesh + hp.w * z_mesh - hfun_c

        # 初期の定常分布を定義
        sd_grid = np.full(hfun_aprime.shape, 1/hfun_aprime.size)

        sd = sd_iteration(sd_grid, hfun_aprime, hp.a_grid, hp.Pz)

        # 4. 総資本供給と総資本需要の差分を計算
        Amesh, _ = np.meshgrid(hp.a_grid, hp.z_grid, indexing='ij')
        A0 = np.sum(Amesh * sd) # 総資本供給
        diff = (A0 - Kd)
        converge_path = np.append(converge_path, diff)
        if DEBUG_MODE:
            print("loop: ", loop)
            print("r0: ", r0, ", A0: ", A0, ", diff: ", diff)

        r0 = r0 -  lambdaR * diff


    r0 = r0 + lambdaR * diff # 最後のループで更新されたr0を使う
    return Result(r_star = r0, w_star = wage, K_star = Kd, 
                hfun_c = hfun_c, hfun_a = hfun_aprime, sd = sd, 
                converge_path = converge_path, loop = loop)