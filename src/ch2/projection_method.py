import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def calc_redidual(
    w: float, theta0, theta1, r: float, beta: float, gamma: float
) -> float:
    a = theta0 + theta1 * w
    if a == 0.0:
        a += 1e-05
    redidual = beta * (1.0 + r) * ((w - a) / ((1.0 + r) * a)) ** gamma - 1.0
    return abs(redidual)


def gen_ssr_func(w_list, r, beta, gamma):
    def calc_ssr(theta) -> float:
        theta0, theta1 = theta
        rss = 0.0
        for w in w_list:
            try:
                rd = calc_redidual(w, theta0, theta1, r, beta, gamma) ** 2
                rss += rd
            except ValueError:
                continue
        return rss

    return calc_ssr


# カリブレーション
beta = 0.985**30
r = 1.025**30 - 1.0
gamma = 2.0

# 状態変数 w のグリッドを生成する
w_list = np.linspace(0.1, 1.0, 10)

# theta0, theta1 をインプットに残差2乗和を計算する関数を定義
calc_ssr = gen_ssr_func(w_list, r, beta, gamma)

theta0_range = np.linspace(0.0, 1.0, 10)
theta1_range = np.linspace(-1.0, 1.0, 10)

# SSR を最小化する theta0, theta1 を見つける
initial_guess = [0.1, 0.1]  # 初期値を設定
result = minimize(calc_ssr, initial_guess, method="Nelder-Mead")

optimal_theta0, optimal_theta1 = result.x
min_ssr = result.fun

print(f"Optimal theta0: {optimal_theta0:.4f}")
print(f"Optimal theta1: {optimal_theta1:.4f}")
print(f"Minimum SSR: {min_ssr:.4f}")

# 3Dプロットのためのデータ作成
theta0_vals, theta1_vals = np.meshgrid(theta0_range, theta1_range)
ssr_vals = np.zeros_like(theta0_vals)

for i in range(theta0_vals.shape[0]):
    for j in range(theta0_vals.shape[1]):
        theta0 = theta0_vals[i, j]
        theta1 = theta1_vals[i, j]
        theta = [theta0, theta1]
        ssr_vals[i, j] = calc_ssr(theta)

# 3Dプロット
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(theta0_vals, theta1_vals, ssr_vals, cmap="viridis")

# 最適解をプロット
ax.scatter(
    optimal_theta0, optimal_theta1, min_ssr, color="red", s=100, label="Optimal point"
)

ax.set_xlabel("theta0")
ax.set_ylabel("theta1")
ax.set_zlabel("SSR")
ax.set_title("3D plot of SSR with respect to theta0 and theta1")
ax.legend()

plt.show()
