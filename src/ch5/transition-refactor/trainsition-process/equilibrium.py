from dataclasses import dataclass
import numpy as np
from policy_function import SolveProblem, TimeIteration
from utils import interpolate_y
from stationary_dist import solve_sd
from setting import Setting

@dataclass
class Equilibrium:
    """ Equilibrium of the model
    """
    Ld: float
    Ls: float
    Kd: float
    Ks: float
    Y: float
    C: float
    r_star: float
    w_star: float
    labor_market_error: float
    goods_market_error: float
    capital_market_error: float

@dataclass
class Result:
    """ Result of equilibrium
    """
    eq: Equilibrium
    hfun_c: np.ndarray
    hfun_a: np.ndarray
    sd: np.ndarray
    converge_path: np.ndarray
    loop: int
    hp: Setting

def search_equilibrium(hp: Setting, lambdaR: float,DEBUG_MODE = False, tol = 1e-6) -> Result:
    """ Search equilibrium
    """


    converge_path = np.empty(0)

    diff = 1
    loop = 0
    while abs(diff) > tol:
        loop += 1

        # ローカル変数を定義
        alpha = hp.alpha #TODO: while文の外で良い
        delta = hp.delta
        r = hp.r
        na = hp.na
        nz = hp.nz
        tau = hp.tau

        # 1. 企業の利潤最大化条件から 総資本需要 K0d, 賃金 wage を求める
        Kd = ((r + delta) / alpha) ** (1 / (alpha - 1)) * hp.Lbar
        wage = (1 - alpha) * ((Kd/hp.Lbar) ** alpha) # 賃金の情報を更新
        hp.w = wage
        hp.Xi = tau * r * Kd # 資本所得税の合計

        # 2. 個人の最適化問題を解いて 政策関数を求める
        hfun_c = SolveProblem(hp,TimeIteration, verbose=DEBUG_MODE)

        # 3. 定常分布を求める
        # hfun_c から 時期のアセット aの政策関数を求める
        hfun_aprime = np.empty((na, nz))
        a_mesh, z_mesh = np.meshgrid(hp.a_grid, hp.z_grid, indexing='ij') # ユニバーサル関数を使用するためのグリッドを生成
        hfun_aprime = (1+(1-tau)*r) * a_mesh + wage * z_mesh - hfun_c

        # 定常分布用のグリッドを用意
        a_grid_sd = np.linspace(-hp.b, hp.a_grid[-1], hp.na_sd)

        # 定常分布の初期値を定義する
        sd_grid = np.full((len(a_grid_sd), nz), 1.0 / (len(a_grid_sd) * nz)) # 各グリッドの初期値を均等に設定
        # sd_grid = np.full((na, nz), 1.0 / (na * nz)) # 各グリッドの初期値を均等に設定
        # sd = sd_iteration(sd_grid, hfun_aprime, hp.a_grid, hp.Pz)
        sd = solve_sd(sd_grid, len(hp.a_grid), len(hp.z_grid), len(a_grid_sd), hp.a_grid, a_grid_sd, hp.Pz, hfun_aprime)
        # sd = solve_sd(sd_grid, na, nz, hp.a_grid, hp.Pz, hfun_aprime)

        # 4. 総資本供給と総資本需要の差分を計算
        Amesh, _ = np.meshgrid(a_grid_sd, hp.z_grid, indexing='ij')
        # 総資本ストック供給 K_s の計算
        Ks = np.sum(Amesh * sd)
        print("Ks: ", Ks)
        
        diff = (Ks - Kd)
        converge_path = np.append(converge_path, diff)
        if DEBUG_MODE:
            print("loop: ", loop)
            print("r: ", r, ", Ks: ", Ks, ", diff: ", diff)

        r = r -  lambdaR * diff
        hp.r = r # r0を更新


    hfun_c_interp = np.empty((len(a_grid_sd), len(hp.z_grid)))
    for iz in range(len(hp.z_grid)):
        interpolate_y(hp.a_grid, a_grid_sd, hfun_c[:, iz], hfun_c_interp[:, iz])

    # 各市場の均衡
    Ld = (hp.w / ((1 - hp.alpha) * Kd ** hp.alpha)) ** (-1 / hp.alpha)
    Labor_market_error = Ld - hp.Lbar
    Y = Kd ** hp.alpha * Ld ** (1 - hp.alpha)
    C = np.sum(hfun_c_interp * sd) 
    goods_market_error = Y - C - hp.delta * Kd
    capital_market_error = diff
    
    hp.r = r + lambdaR * diff # 最後のループで更新されたrを使う
    return Result(eq=Equilibrium(Ld=Ld, Ls=hp.Lbar, Kd=Kd, Ks=Ks, Y=Y, C=C,
                r_star=hp.r, w_star=hp.w,
                labor_market_error=Labor_market_error,
                goods_market_error=goods_market_error,
                capital_market_error=capital_market_error),
                hfun_c=hfun_c, hfun_a=hfun_aprime, sd=sd,
                converge_path=converge_path, loop=loop, hp=hp)