from numba import njit
import numpy as np

# 線形補間を効率的に行うコード（interpolationパッケージのinterpでは正しい結果が得られなかった）
# I took this function from state-space Jacobian package
# @guvectorize(['void(float64[:], float64[:], float64[:], float64[:])'], '(n),(nq),(n)->(nq)')
@njit
def interpolate_y(x, xq, y, yq):
    """Efficient linear interpolation exploiting monotonicity.
    Complexity O(n+nq), so most efficient when x and xq have comparable number of points.
    Extrapolates linearly when xq out of domain of x.
    Parameters
    ----------
    x  : array (n), ascending data points
    xq : array (nq), ascending query points
    y  : array (n), data points
    Returns
    ----------
    yq : array (nq), interpolated points
    """
    nxq, nx = xq.shape[0], x.shape[0]

    xi = 0
    x_low = x[0]
    x_high = x[1]
    for xqi_cur in range(nxq):
        xq_cur = xq[xqi_cur]
        while xi < nx - 2:
            if x_high >= xq_cur:
                break
            xi += 1
            x_low = x_high
            x_high = x[xi + 1]

        xqpi_cur = (x_high - xq_cur) / (x_high - x_low)
        yq[xqi_cur] = xqpi_cur * y[xi] + (1 - xqpi_cur) * y[xi + 1]

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
                        na_sd: int,    # mu0.shape[0]のこと
                        agrid: np.ndarray,
                        agrid_sd: np.ndarray,
                        Pz: np.ndarray,
                        pfgrid: np.ndarray,
                        critmu: float = 1e-6):

    # -1. 政策関数の行列を補間して、行数を na 個から na_sd 個に変える
    # pfgrid -> pfgrid_new
    pfgrid_new = np.empty((na_sd,nz))              # 空の箱を用意
    for z_index in range(nz):
        aprime_new = np.empty(na_sd)
        interpolate_y(agrid, agrid_sd, pfgrid[:, z_index], aprime_new)
        pfgrid_new[:,z_index] = aprime_new


    # 0. transition matrixを計算
    G = np.zeros((na_sd*nz,na_sd*nz))
    weight = np.zeros((na_sd,nz))

    for iz in range(nz):
        for ia in range(na_sd):

            # 政策関数 pfgrid[ia, iz] に合わせる
            aprime = pfgrid_new[ia, iz]


            if aprime < agrid_sd[0]:
                weight[ia, iz] = 1.0
                jtilde = 0
            elif aprime > agrid_sd[-1]:
                weight[ia, iz] = 0
                jtilde = na_sd - 2
            else:
                # aprimeが落ちる区間の左端のグリッドのインデックスを探す
                jtilde = gridlookup2(aprime, agrid_sd)
                # 重み weight[ia, iz] に格納
                weight[ia, iz] = (agrid_sd[jtilde+1] - aprime) / (agrid_sd[jtilde+1] - agrid_sd[jtilde])

            # スタックしたときの現在の状態のインデックスを計算
            i_s = iz * na_sd + ia

            # 次期外生状態 jz でループ
            for jz in range(nz):

                # スタックしたときの次期の状態のインデックスを計算
                j_s  = jz * na_sd + jtilde

                # 遷移確率行列 G に代入
                #   - Pe[iz, jz] : 現在 iz から次期 jz への外生ショックの遷移確率
                G[i_s, j_s ] += weight[ia, iz]   * Pz[iz, jz]
                G[i_s, j_s + 1] += (1.0 - weight[ia, iz]) * Pz[iz, jz]


    diffmu = 1.0e4
    mu1    = np.zeros((na_sd, nz))

    # 分布の初期値をスタックしてベクトル化
    mu0 = mu0.T.flatten()

    while diffmu > 1e-6:

        # 1. 分布の更新 : dist_new = G' * dist
        mu1 = G.T @ mu0

        # 2. 収束判定
        diffmu = np.max(np.abs(mu1 - mu0))

        # 3. normalizeして次のループへ
        mu0 = mu1 / np.sum(mu1)

    sd = mu0.reshape(nz, na_sd).T
    return sd