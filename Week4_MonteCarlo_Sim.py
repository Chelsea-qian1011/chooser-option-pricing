import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import config


# ================= 1. BSM 定价公式 =================
def bs_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


# ================= 2. Heston 随机波动率蒙特卡洛 =================
# 参数设置
np.random.seed(42)
T2 = config.T2
N_steps = 252  # 假设一年 252 个交易日
dt = T2 / N_steps
S0 = config.S
K = config.K
r = config.r

# Heston 模型参数
v0 = 0.04  # 初始方差 (对应 20% 波动率)
kappa = 2.0  # 均值回归速度
theta = 0.04  # 长期方差
sigma_v = 0.3  # 波动率的波动率 (Vol of Vol)
rho = -0.7  # 资产与波动的相关性 (杠杆效应: 股价跌，波动升)

# 模拟路径数量
M = 1000  # 1000 条路径
S_paths = np.zeros((M, N_steps + 1))
v_paths = np.zeros((M, N_steps + 1))

S_paths[:, 0] = S0
v_paths[:, 0] = v0

for t in range(1, N_steps + 1):
    # 生成相关的布朗运动
    Z1 = np.random.normal(0, 1, M)
    Z2 = np.random.normal(0, 1, M)
    W1 = Z1
    W2 = rho * Z1 + np.sqrt(1 - rho ** 2) * Z2

    # Heston 更新方差 (Feller 条件确保方差不为负)
    v_prev = np.maximum(v_paths[:, t - 1], 0)
    v_paths[:, t] = v_prev + kappa * (theta - v_prev) * dt + sigma_v * np.sqrt(v_prev * dt) * W2
    v_paths[:, t] = np.maximum(v_paths[:, t], 0)  # 防止负数

    # 更新股票价格
    S_paths[:, t] = S_paths[:, t - 1] * np.exp((r - 0.5 * v_prev) * dt + np.sqrt(v_prev * dt) * W1)

# 计算蒙特卡洛模拟的期权价格 (到期日 T2)
payoffs = np.maximum(S_paths[:, -1] - K, 0)
mc_price_raw = np.exp(-r * T2) * np.mean(payoffs)

# ================= 3. 加入真实市场摩擦 (Friction) =================
# 3.1 加入跳空 (Jump)
jump_prob = 0.05
jump_size_mean = -0.02
jump_size_std = 0.05
jumps = np.random.binomial(1, jump_prob, M) * np.random.normal(jump_size_mean, jump_size_std, M)
payoffs_jump = np.maximum(S_paths[:, -1] * np.exp(jumps) - K, 0)
mc_price_jump = np.exp(-r * T2) * np.mean(payoffs_jump)

# 3.2 加入买卖价差 (Bid-Ask Spread) 摩擦
spread = 0.05  # 假设 5% 的摩擦成本
market_price_with_friction = mc_price_jump + spread

# ================= 4. 计算 BSM 误差并输出 =================
# BSM 的预测价格 (用常数波动率 20%)
bsm_price = bs_call(S0, K, T2, r, config.sigma)

# 计算 MAE 和 RMSE
# 这里我们将蒙特卡洛模拟出的带有摩擦的市场价作为 "Actual Price"
actual_price = market_price_with_friction
mae = abs(actual_price - bsm_price)
rmse = np.sqrt((actual_price - bsm_price) ** 2)

print("===== Advanced Monte Carlo Simulation (Heston + Jumps + Frictions) =====")
print(f"BSM Constant Volatility Price: ${bsm_price:.4f}")
print(f"Simulated Market Price (Heston + Jumps + Spread): ${actual_price:.4f}")
print("-------------------------------------------------")
print(f"Mean Absolute Error (MAE):  ${mae:.4f}")
print(f"Root Mean Squared Error (RMSE): ${rmse:.4f}")
print("-------------------------------------------------")
print(
    "Note: The BSM model fails to capture random volatility (Heston), jumps, and bid-ask spreads, leading to underpricing/overpricing in real markets.")