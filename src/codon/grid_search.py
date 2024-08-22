# grid searchについてのコードです。
# Codon と Cpython 両方で実行できます。
from time import time


def generate_grid(start: float, end: float, interval: float):
    # python の標準パッケージのみで実装
    grid = []
    i = start
    while i <= end:
        grid.append(i)
        i += interval
    return grid


def f(w: float, a: float, r: float, beta: float, gamma: float) -> float:
    if (1 - gamma) == 0:
        return float("inf")  # ゼロ割を避けるために inf を返す
    if (w - a) < 0:
        return float("-inf")  # 無限に小さな値を返す
    try:
        # CRRA型効用関数
        u_young = ((w - a)**(1 - gamma)) / (1 - gamma)
        u_old = beta * ((r * a)**(1 - gamma)) / (1 - gamma)
        return u_young + u_old
    except ZeroDivisionError:
        return float("-inf")


def main():
    # カリブレーション
    beta = 0.985 ** 30
    r = 1.025 ** 30
    gamma: float = 2.0
    # print(beta, r, gamma)
    # 状態変数 w のグリッドを生成する
    w = generate_grid(0.1, 1.0, 0.1)
    # 制御変数 a のグリッドを生成する
    a = generate_grid(0.025, 1.0, 0.025)
    # print(w, a)

    policy_function = []  # 最適な貯蓄aの配列
    for i in range(len(w)):
        u_list = []

        for j in range(len(a)):
            u_list.append(f(w[i], a[j], r, beta, gamma))

        policy_function.append(a[u_list.index(max(u_list))])
    return policy_function


t0 = time()
pf = main()
t1 = time()
print(f'Computed policy function = \n\n{pf}\n\nin {t1 - t0} seconds.')
