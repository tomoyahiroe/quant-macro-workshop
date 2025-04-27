import numpy as np
from setting import Setting
from equilibrium import search_equilibrium
import time

loops = 1
times = np.empty(loops)
r_stars = np.empty(loops)
k_star = np.empty(loops)
w_star = np.empty(loops)
for i in range(loops):
    start = time.time()
    hp = Setting(beta=0.96, gamma=3, rho=0.6, sigma=0.4, 
                alpha=0.36, delta=0.08, b=3, a_max=45,
                nz = 7, na = 300, na_sd=800, r0 = 0.03, tau = 0.0) # brent法でエラーが出ないようにwの初期値を0以外に設定
    result = search_equilibrium(hp = hp, lambdaR = 0.002, DEBUG_MODE= True)
    end = time.time()

    times[i] = (end - start)
    r_stars[i] = result.r_star
    k_star[i] = result.K_star
    w_star[i] = result.w_star
    

print(f"TIME: {np.mean(times)}({np.std(times)})")
print(f"r_star: {np.mean(result.r_star)}({np.std(result.r_star)})")
print(f"K_star: {np.mean(result.K_star)}({np.std(result.K_star)})")
print(f"w_star: {np.mean(result.w_star)}({np.std(result.w_star)})")
