import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
from scipy.stats import norm
import config

# ================= 1. 准备数据 =================
print("加载特征数据...")
df = pd.read_csv("features.csv")
df['Date'] = pd.to_datetime(df['Date'])

# 目标变量（Target）：明天的真实 BSM 期权价格
# 我们先算一个基于今天历史波动率的标准 BSM 价格，作为模型预测的基准
def bs_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

# 用今天的 Vol_20 算一个理论价，明天挪到 Target 列
df['Target_Price'] = df['Vol_20'].apply(lambda v: bs_call(config.S, config.K, config.T2, config.r, v))
df['Target_Price'] = df['Target_Price'].shift(-1)
df = df.dropna().reset_index(drop=True)

# 挑选特征（X）
feature_cols = ['Return', 'Vol_5', 'Vol_20', 'Vol_60', 'Volume_MA5', 'VIX_Change', 'VIX_JPM_Corr', 'Rate_Momentum', 'Return_Lag1', 'MA_Gap']
X = df[feature_cols]
y = df['Target_Price']

# ================= 2. 时间序列切分 (70% / 15% / 15%) =================
train_size = int(len(df) * 0.70)
val_size = int(len(df) * 0.15)

X_train, y_train = X.iloc[:train_size], y.iloc[:train_size]
X_val, y_val = X.iloc[train_size : train_size+val_size], y.iloc[train_size : train_size+val_size]
X_test, y_test = X.iloc[train_size+val_size:], y.iloc[train_size+val_size:]

print(f"数据切分完成: 训练集 {len(X_train)}天, 验证集 {len(X_val)}天, 测试集 {len(X_test)}天")

# ================= 3. 训练多个模型 =================
print("\n正在训练端到端定价模型...")

# 模型 A: 线性回归 (Linear Regression)
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)

# 模型 B: GBDT (梯度提升树)
gbdt_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)
gbdt_model.fit(X_train, y_train)
gbdt_preds = gbdt_model.predict(X_test)

# ================= 4. 输出结果对比 =================
print("\n===== Approach 2: End-to-End Supervised Pricing =====")
print(f"测试集真实价格均值: ${y_test.mean():.4f}")
print(f"线性回归模型预测均价: ${lr_preds.mean():.4f}")
print(f"GBDT 模型预测均价:    ${gbdt_preds.mean():.4f}")

# 把结果存下来
results_df = pd.DataFrame({
    'Date': df['Date'].iloc[train_size+val_size:],
    'Actual_Price': y_test,
    'Pred_LR': lr_preds,
    'Pred_GBDT': gbdt_preds
})
results_df.to_csv("Week5_Approach2_Results.csv", index=False)
print("\n预测结果已保存至 Week5_Approach2_Results.csv")