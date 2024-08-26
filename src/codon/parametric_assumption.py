from scipy.optimize import minimize
from time import time


def generate_grid(start: float, end: float, interval: float):
    # python の標準パッケージのみで実装
    grid = []
    i = start
    while i <= end:
        grid.append(i)
        i += interval
    return grid


def calc_residual(w: float, theta0: float, theta1: float,
                  r: float, beta: float, gamma: float) -> float:
    a = theta0 + theta1 * w
    epsilon = 1e-8  # 小さな正の数を加えることで不安定性を回避
    if w - a <= epsilon:
        return float('inf')
    residual = beta * (1.0 + r) * ((1.0 + r) * a +
                                   epsilon)**(-gamma) / (w - a + epsilon)**(-gamma) - 1.0
    return abs(residual)


def gen_rss_func(w_list, r, beta, gamma):
    def calc_rss(theta: [float]) -> float:
        theta0, theta1 = theta
        rss = 0.0
        for w in w_list:
            rss += calc_residual(w, theta0, theta1, r, beta, gamma)**2.0
        return rss
    return calc_rss


def main():
    # カリブレーション
    beta = 0.985 ** 30
    r = 1.025 ** 30 - 1.0
    gamma: float = 2.0

    # 状態変数 w のグリッドを生成する
    w_list = generate_grid(0.1, 1.0, 0.1)

    # theta0, theta1 をインプットに残差2乗和を計算する関数を定義
    calc_rss = gen_rss_func(w_list, r, beta, gamma)

    # result = calc_rss([0.0, 0.35])

    result = minimize(calc_rss, x0=[0.5, 0.5],
                      method="trust-constr",
                      bounds=[(0.0, 1.0), (0.0, 1.0)],
                      options={'maxiter': 10000})
    return result


t0 = time()
pf = main()
t1 = time()
print(f'Computed policy function = \n\n{pf}\n\nin {t1 - t0} seconds.')
