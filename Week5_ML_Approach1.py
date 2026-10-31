import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from scipy.stats import norm
import config

# ================= 1. BSM 定价公式 =================
def bs_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

# ================= 2. 读取第二周准备好的特征数据 =================
print("加载特征数据...")
df = pd.read_csv("features.csv")
df['Date'] = pd.to_datetime(df['Date'])

# 我们的目标是预测明天的波动率。
# 把 Vol_20 往下挪一天作为目标（Target），也就是用今天的信息预测明天的波动。
df['Target_Next_Vol'] = df['Vol_20'].shift(-1)
df = df.dropna().reset_index(drop=True)

# 挑选特征（X）和目标（y）
feature_cols = ['Return', 'Vol_5', 'Vol_60', 'Volume_MA5', 'VIX_Change', 'VIX_JPM_Corr', 'Rate_Momentum', 'Return_Lag1', 'MA_Gap']
X = df[feature_cols]
y = df['Target_Next_Vol']

# ================= 3. 时间序列切分 (70% / 15% / 15%) =================
# 关键点：绝对不能打乱顺序（不能 shuffle），必须按时间切！防止前视偏差！
train_size = int(len(df) * 0.70)
val_size = int(len(df) * 0.15)

X_train, y_train = X.iloc[:train_size], y.iloc[:train_size]
X_val, y_val = X.iloc[train_size : train_size+val_size], y.iloc[train_size : train_size+val_size]
X_test, y_test = X.iloc[train_size+val_size:], y.iloc[train_size+val_size:]

print(f"数据切分完成: 训练集 {len(X_train)}天, 验证集 {len(X_val)}天, 测试集 {len(X_test)}天")

# ================= 4. 训练机器学习模型 =================
print("\n正在训练 XGBoost 模型...")
model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

# 在测试集上预测未来的波动率
predicted_vol = model.predict(X_test)

# ================= 5. 将预测的波动率喂给 BSM 算价格 =================
predicted_prices = []
for vol in predicted_vol:
    # 用 ML 预测出的波动率来算期权价格（加个 max 防止预测出负波动率）
    price = bs_call(config.S, config.K, config.T2, config.r, max(vol, 0.01))
    predicted_prices.append(price)

# ================= 6. 输出初步结果 =================
print("\n===== Approach 1: ML Volatility Prediction + BSM =====")
print(f"真实历史波动率均值: {y_test.mean():.4f}")
print(f"模型预测波动率均值: {np.mean(predicted_vol):.4f}")
print(f"基于预测波动率的 BSM 期权平均价格: ${np.mean(predicted_prices):.4f}")

# 把预测结果存下来，后面算误差用
results_df = pd.DataFrame({
    'Date': df['Date'].iloc[train_size+val_size:],
    'Actual_Vol': y_test,
    'Predicted_Vol': predicted_vol,
    'BSM_Price': predicted_prices
})
results_df.to_csv("Week5_Approach1_Results.csv", index=False)
print("\n预测结果已保存至 Week5_Approach1_Results.csv")
