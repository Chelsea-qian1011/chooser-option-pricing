import numpy as np
import pandas as pd
from scipy.stats import norm
import config


# 1. 基准 BSM 公式
def bs_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


# 2. 模拟的“真实市场”基准价格（用我们昨天跑出的 Heston+摩擦 结果）
# 昨天跑出来的是 $15.1180
benchmark_price = 15.1180

# 3. 设置我们要测试的多个波动率
test_sigmas = [0.10, 0.15, 0.20, 0.25, 0.30]

results = []

# 4. 循环计算不同 Sigma 下的 BSM 价格和误差
for sigma in test_sigmas:
    # 算 BSM 价格
    bsm_price = bs_call(config.S, config.K, config.T2, config.r, sigma)

    # 算误差
    mae = abs(bsm_price - benchmark_price)
    rmse = np.sqrt((bsm_price - benchmark_price) ** 2)

    # 存入列表
    results.append({
        "Volatility (Sigma)": f"{sigma * 100:.0f}%",
        "BSM Price": round(bsm_price, 4),
        "Benchmark Price": round(benchmark_price, 4),
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4)
    })

# 5. 打印对照表 (Markdown 格式)
df = pd.DataFrame(results)
print("===== BSM Volatility Ablation Study =====")
print(df.to_markdown(index=False))

# 6. 自动把这个表格保存为 CSV 文件，方便你放进 Word
df.to_csv("Week4_Ablation_Table.csv", index=False)
print("\n表格已保存为 Week4_Ablation_Table.csv")