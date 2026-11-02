import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from scipy.stats import norm
import config

# 1. 评估指标计算函数
def evaluate_model(y_true, y_pred, model_name):
    # 强行转为 numpy 数组计算，绝对不会有长度对齐问题
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return [model_name, round(mae, 4), round(rmse, 4), round(r2, 4), round(mape, 4)]

# 2. 加载数据
df = pd.read_csv("features.csv")
df['Date'] = pd.to_datetime(df['Date'])
df['ATM_IV'] = df['VIX'] / 100.0

# 特征列
feature_cols = ['Return', 'Vol_5', 'Vol_20', 'Vol_60', 'Volume_MA5', 'VIX_Change', 'VIX_JPM_Corr', 'Rate_Momentum', 'Return_Lag1', 'MA_Gap']

# ================= Approach 1: 预测 Implied Vol =================
print("计算 Approach 1 (预测 ATM IV)...")
X = df[feature_cols].values  # 转 numpy
y_iv = df['ATM_IV'].values

train_size = int(len(df) * 0.70)
val_size = int(len(df) * 0.15)

X_train_iv, X_test_iv = X[:train_size], X[train_size+val_size:]
y_train_iv, y_test_iv = y_iv[:train_size], y_iv[train_size+val_size:]

xgb_iv = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)
xgb_iv.fit(X_train_iv, y_train_iv)
pred_iv = xgb_iv.predict(X_test_iv)

res_iv = evaluate_model(y_test_iv, pred_iv, "Approach 1 (XGBoost ATM IV)")

# ================= Approach 2: BSM + ML 残差学习 =================
print("计算 Approach 2 (BSM + ML 残差)...")
def bs_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

# 用真实 IV 算出 BSM 基准价
df['BSM_Base'] = df['ATM_IV'].apply(lambda v: bs_call(config.S, config.K, config.T2, config.r, v))

# 残差 = 今天的 BSM 价 - 明天的 BSM 价
df['Residual'] = df['BSM_Base'] - df['BSM_Base'].shift(-1)

# 删除 NaN 后，将数据转成纯 numpy 数组
df_clean = df.dropna().copy()
X_clean = df_clean[feature_cols].values
y_resid = df_clean['Residual'].values
y_base = df_clean['BSM_Base'].values

# 重新切分（用干净的数组切）
train_size = int(len(df_clean) * 0.70)
val_size = int(len(df_clean) * 0.15)

X_train_res, X_test_res = X_clean[:train_size], X_clean[train_size+val_size:]
y_train_res = y_resid[:train_size]
y_test_res = y_resid[train_size+val_size:]
y_test_base = y_base[train_size+val_size:]

# 训练 GBDT 预测残差
gbdt_resid = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)
gbdt_resid.fit(X_train_res, y_train_res)
pred_resid = gbdt_resid.predict(X_test_res)

# 最终预测 = 真实 BSM 基础价 + 预测出的残差
# 现在全都是纯 numpy 数组，长度绝对一致！
pred_final = y_test_base + pred_resid
res_resid = evaluate_model(y_test_base, pred_final, "Approach 2 (BSM + GBDT Residual)")

# ================= 输出对比 =================
results = pd.DataFrame([res_iv, res_resid], columns=['Model', 'MAE', 'RMSE', 'R2', 'MAPE'])
print("\n===== 实验评估对比 =====")
print(results.to_markdown(index=False))