import TI_EGM_stochastic as Model
import matplotlib.pyplot as plt
import numpy as np

# インスタンスを生成
parameters = Model.Parameters(delta = 0.5)
model = Model.Model(parameters)

model.solve()

# 結果を表示

## Kのグリッドを生成
k_grid = np.linspace(parameters.kp_min, parameters.kp_max, parameters.kp_size)
k_matrix = np.array(list(k_grid) * len(parameters.A)).reshape(parameters.kp_size, len(parameters.A))

## Mの行列を生成
prod_func = Model.ProductionFunc()
wealth_func = Model.WealthFunc(prod_func, delta=parameters.delta)
m = np.zeros((parameters.kp_size, len(parameters.A)))
for i, kp in enumerate(k_grid):
    for j, a in enumerate(parameters.A):
        m[i, j] = wealth_func.wealth(kp, a)

## Aの行列を生成
A_matrix = np.array([parameters.A] * parameters.kp_size)

## 政策関数をユニバーサル関数化
np_pf = np.frompyfunc(model.policy_func, 2, 1)

## 3D plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(m, A_matrix, np_pf(m, A_matrix), cmap='bwr')
# ax.set_xlabel('M')
# ax.set_ylabel('A')
# ax.set_zlabel('M')
# plt.show()

## K, K'の関係をプロット
kp = np.zeros((parameters.kp_size, len(parameters.A)))
kp = m - np_pf(m, A_matrix)
fig, ax = plt.subplots()
for i in range(len(parameters.A)):
    ax.plot(k_grid, kp[:, i], label=f'A = {parameters.A[i]}')
ax.legend()
ax.set_xlabel('K')
ax.set_ylabel('K\'')
plt.show()
