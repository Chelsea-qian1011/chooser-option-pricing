# ========== 导入工具 ==========
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ========== 读取特征文件 ==========
df = pd.read_csv("features.csv", parse_dates=["Date"])

# ========== 图1：四张时间序列子图 ==========
fig, axes = plt.subplots(4, 1, figsize=(14, 14))

# 第1张：JPM 收盘价
axes[0].plot(df["Date"], df["Close"], color="blue")
axes[0].set_title("JPM Close Price (2018-2024)")
axes[0].set_ylabel("Price ($)")
axes[0].grid(True, alpha=0.3)

# 第2张：VIX 恐慌指数
axes[1].plot(df["Date"], df["VIX"], color="red")
axes[1].set_title("VIX Index")
axes[1].set_ylabel("VIX")
axes[1].grid(True, alpha=0.3)

# 第3张：滚动波动率
axes[2].plot(df["Date"], df["Vol_20"], label="Vol_20 (middle)", color="orange")
axes[2].plot(df["Date"], df["Vol_60"], label="Vol_60 (long)", color="green")
axes[2].set_title("Rolling Volatility (Annualized)")
axes[2].set_ylabel("Volatility")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

# 第4张：国债利率
axes[3].plot(df["Date"], df["DGS10"], color="purple")
axes[3].set_title("US 10-Year Treasury Yield")
axes[3].set_ylabel("Yield (%)")
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("overview.png", dpi=150)
plt.show()

# ========== 图2：相关性热力图 ==========
plt.figure(figsize=(14, 12))
sns.heatmap(
    df.drop(columns=["Date"]).corr(),
    cmap="coolwarm",
    center=0,
    annot=True,
    fmt=".2f",
    square=True
)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation.png", dpi=150)
plt.show()

print("图表已保存：overview.png, correlation.png")