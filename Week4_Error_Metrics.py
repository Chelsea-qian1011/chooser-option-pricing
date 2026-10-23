import numpy as np
import pandas as pd
from scipy.stats import norm
import config

# ====== 1. 核心 BSM 定价公式 ======
def bs_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

# ====== 2. 模拟“真实的”市场交易数据 ======
# 因为我们暂时没接真实 CME 接口，这里先模拟 5 天真实市场的数据来测试误差计算器
# （实际市场价格会围绕理论价格上下波动，因为有买卖价差和市场摩擦）
np.random.seed(42)
true_market_prices = []
model_prices = []

for i in range(5):
    # 模拟市场价格（理论价 $20.77 上下浮动 5% 到 10%）
    noise = np.random.uniform(0.95, 1.05)
    theoretical_price = bs_call(config.S, config.K, config.T2, config.r, config.sigma) # 这里用普通 Call 测试
    model_prices.append(theoretical_price)
    true_market_prices.append(theoretical_price * noise + np.random.normal(0, 0.5))

# ====== 3. 计算误差指标 ======
mae = np.mean(np.abs(np.array(true_market_prices) - np.array(model_prices)))
rmse = np.sqrt(np.mean((np.array(true_market_prices) - np.array(model_prices))**2))

print("===== Baseline Model Performance Evaluation =====")
print(f"Model Price (BSM): ${model_prices[0]:.4f}")
print(f"Simulated Market Prices: {[round(p, 2) for p in true_market_prices]}")
print("-------------------------------------------------")
print(f"Mean Absolute Error (MAE):  ${mae:.4f}")
print(f"Root Mean Squared Error (RMSE): ${rmse:.4f}")
print("-------------------------------------------------")
print("Note: These are simulated errors for pipeline testing. Real CME data will replace this in the final report.")