from numba import njit
import numpy as np
from numba import guvectorize, float64
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