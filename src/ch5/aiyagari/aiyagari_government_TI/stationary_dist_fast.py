from numba import njit
import numpy as np

# x0が存在する区間の左端のグリッドを見つける関数
@njit
def gridlookup2(x0, xgrid):
    nx = np.shape(xgrid)[0]
    ix = 0
    for jx in range(nx):
        if x0 <= xgrid[jx]:
            break
        ix += 1
    ix = min(max(1, ix), nx-1)
    return ix - 1


# 定常分布を求める関数
@njit
def solve_sd(mu0: np.ndarray, 
            na: int, 
            nz: int, 
            agrid: np.ndarray,
            Pz: np.ndarray,
            pfgrid: np.ndarray):
                        
    # 0. transition matrixを計算
    G = np.zeros((na*nz,na*nz))
    weight = np.zeros((na,nz))

    for iz in range(nz):
        for ia in range(na):
        
            # 政策関数 pfgrid[ia, iz] に合わせる
            aprime = pfgrid[ia, iz]

            
            if aprime < agrid[0]:
                weight[ia, iz] = 1.0
                jtilde = 0
            elif aprime > agrid[-1]:
                weight[ia, iz] = 0
                jtilde = na - 2
            else:
                # aprime が落ちる区間の左端のグリッドのインデックスを探す
                jtilde = gridlookup2(aprime, agrid)
                # 重み weight[ia, iz] に格納
                weight[ia, iz] = (agrid[jtilde+1] - aprime) / (agrid[jtilde+1] - agrid[jtilde])

            # スタックしたときの現在の状態のインデックスを計算
            i_s = iz * na + ia

            # 次期外生状態 jz でループ
            for jz in range(nz):

                # スタックしたときの次期の状態のインデックスを計算
                j_s  = jz * na + jtilde    

                # 遷移確率行列 G に代入
                #   - Pz[iz, jz] : 現在 iz から次期 jz への外生ショックの遷移確率
                G[i_s, j_s ] += weight[ia, iz]   * Pz[iz, jz]
                G[i_s, j_s + 1] += (1.0 - weight[ia, iz]) * Pz[iz, jz]

    
    # 1. iterationで定常分布を求める
    diffmu = 1.0e4
    mu1    = np.zeros((na, nz))

    # 分布の初期値をスタックしてベクトル化
    mu0 = mu0.T.flatten()

    while diffmu > 1e-6:

        # 1. 分布の更新 : dist_new = G' * dist
        mu1 = G.T @ mu0

        # 2. 収束判定
        diffmu = np.max(np.abs(mu1 - mu0))

        # 3. normalizeして次のループへ
        mu0 = mu1 / np.sum(mu1)
    
    sd = mu0.reshape(nz, na).T
    return sd