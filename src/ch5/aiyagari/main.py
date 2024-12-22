import numpy as np
from setting import Setting
from equilibrium import search_equilibrium
import time


# 時間を計測
start = time.time()
result = search_equilibrium(hp = Setting(beta=0.96, gamma=3, rho=0.6, sigma=0.4, alpha=0.36, delta=0.08, b=3, lambdaR = 0.002), 
                            DEBUG_MODE= True)
end = time.time()
print(f"TIME: {end - start}")