import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import config

# 1. BSM 定价公式
def bs_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

# 基础参数
S = config.S
K = config.K
T = config.T2

# ================= 极端场景 1：波动率飙升 50% =================
vol_scenarios = [0.20, 0.30, 0.50] # 20%, 30%, 50%
prices_vol = [bs_call(S, K, T, config.r, v) for v in vol_scenarios]

# ================= 极端场景 2：利率上调 2% =================
rate_scenarios = [0.05, 0.07, 0.09] # 5%, 7%, 9%
prices_rate = [bs_call(S, K, T, r, config.sigma) for r in rate_scenarios]

# ================= 画出对比图 =================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 图1：波动率变化对期权价格的影响
axes[0].bar(["Vol=20%", "Vol=30%", "Vol=50%"], prices_vol, color=['blue', 'orange', 'red'])
axes[0].set_title("Stress Test: Impact of Volatility Spike")
axes[0].set_ylabel("Option Price ($)")
for i, v in enumerate(prices_vol):
    axes[0].text(i, v + 0.5, f"${v:.2f}", ha='center')

# 图2：利率变化对期权价格的影响
axes[1].bar(["Rate=5%", "Rate=7%", "Rate=9%"], prices_rate, color=['green', 'purple', 'brown'])
axes[1].set_title("Stress Test: Impact of Interest Rate Hike")
axes[1].set_ylabel("Option Price ($)")
for i, v in enumerate(prices_rate):
    axes[1].text(i, v + 0.5, f"${v:.2f}", ha='center')

plt.tight_layout()
plt.savefig("Week7_Stress_Test.png", dpi=150)
plt.show()

print("压力测试完成！图表已保存为 Week7_Stress_Test.png")
print(f"波动率 20% -> 50% 时，期权价格从 ${prices_vol[0]:.2f} 涨到 ${prices_vol[2]:.2f}")
print(f"利率 5% -> 9% 时，期权价格从 ${prices_rate[0]:.2f} 变到 ${prices_rate[2]:.2f}")