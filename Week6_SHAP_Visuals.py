import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from scipy.stats import norm
import config

# 准备数据
df = pd.read_csv("features.csv")
df['ATM_IV'] = df['VIX'] / 100.0
feature_cols = ['Return', 'Vol_5', 'Vol_20', 'Vol_60', 'Volume_MA5', 'VIX_Change', 'VIX_JPM_Corr', 'Rate_Momentum', 'Return_Lag1', 'MA_Gap']
X = df[feature_cols].values

# 计算期权真实价格作为 Target (而不是残差)
def bs_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

df['Target_Price'] = df['ATM_IV'].apply(lambda v: bs_call(config.S, config.K, config.T2, config.r, v))
df_clean = df.dropna().copy()
X_clean = df_clean[feature_cols].values
y_price = df_clean['Target_Price'].values

# 训练新模型
train_size = int(len(df_clean) * 0.70)
X_train, X_test = X_clean[:train_size], X_clean[train_size:]
y_train, y_test = y_price[:train_size], y_price[train_size:]

model = GradientBoostingRegressor(n_estimators=50, learning_rate=0.01, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# 画 SHAP 图
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, feature_names=feature_cols, show=False)
plt.title("SHAP Feature Importance (Direct Option Pricing)", fontsize=14)
plt.tight_layout()
plt.savefig("Week6_SHAP_Summary_V2.png", bbox_inches='tight', dpi=150)
print("新的 SHAP 图已生成：Week6_SHAP_Summary_V2.png")