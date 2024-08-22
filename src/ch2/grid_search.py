# ライブラリのインポート
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# 状態変数 w のグリッドを生成
w_grid = np.linspace(0.1, 1.0, 10)
# 制御変数 a のグリッドを生成
a_grid = np.linspace(0.025, 1.0, 40)


# パラメータを設定 (カリブレーション)
beta = 0.985**30
gamma = 2
r = (1.025**30) - 1


# CRRA型効用関数を定義する
def vf(w, a, beta, gamma, r):
    """
    消費が負となる選択をした場合効用がマイナス無限大になるように設定して、
    そのような選択肢が選ばれないようにする。
    """
    if w - a < 0:
        return float("-inf")
    u_young = ((w - a) ** (1 - gamma)) / (1 - gamma)
    u_old = (((1 + r) * a) ** (1 - gamma)) / (1 - gamma)
    u = u_young + beta * u_old
    return u


# a_starの初期化（格納する箱をつくる）
a_box = []
# グリッドを代入して政策関数を求める
for w in w_grid:
    # v_boxをつくる
    v_box = []  # v_boxの初期化
    for a in a_grid:
        util = vf(w, a, beta, gamma, r)
        v_box.append(util)
    a_star = a_grid[np.argmax(v_box)]
    a_box.append(a_star)


# 最適なwとaの関係をグラフで描画する
df = pd.DataFrame({"x_axis": w_grid, "y_axis": a_box})

# plot
plt.plot("x_axis", "y_axis", data=df, linestyle="-", marker="o")
plt.show()
