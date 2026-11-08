import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from scipy.stats import norm
import shap
import pickle
import config

# ================= 1. 准备数据 =================
print("加载数据...")
df = pd.read_csv("features.csv")
df['ATM_IV'] = df['VIX'] / 100.0
feature_cols = ['Return', 'Vol_5', 'Vol_20', 'Vol_60', 'Volume_MA5', 'VIX_Change', 'VIX_JPM_Corr', 'Rate_Momentum', 'Return_Lag1', 'MA_Gap']
X = df[feature_cols].values

# 算 BSM 基准和残差
def bs_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

df['BSM_Base'] = df['ATM_IV'].apply(lambda v: bs_call(config.S, config.K, config.T2, config.r, v))
df['Residual'] = df['BSM_Base'] - df['BSM_Base'].shift(-1)

# 清洗并对齐数据
df_clean = df.dropna().copy()
X_clean = df_clean[feature_cols].values
y_resid = df_clean['Residual'].values
y_base = df_clean['BSM_Base'].values

# 时间序列切分 (70% 训练，30% 测试，这里为了简洁省去验证集，把15%合并进训练)
train_size = int(len(df_clean) * 0.70)
X_train, X_test = X_clean[:train_size], X_clean[train_size:]
y_train_res, y_test_res = y_resid[:train_size], y_resid[train_size:]
y_test_base = y_base[train_size:]

# ================= 2. 超参数调优 (Hyperparameter Optimization) =================
print("正在使用 TimeSeriesSplit 进行网格搜索调优...")
# 定义要搜索的参数网格
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 4, 5]
}

# 时间序列交叉验证（防止未来数据泄露）
tscv = TimeSeriesSplit(n_splits=3)
gbdt = GradientBoostingRegressor(random_state=42)
grid_search = GridSearchCV(estimator=gbdt, param_grid=param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train_res)

best_model = grid_search.best_estimator_
print(f"最佳参数: {grid_search.best_params_}")

# ================= 3. 模型评估与对比 =================
print("\n正在测试集上进行预测...")
pred_resid = best_model.predict(X_test)
pred_final = y_test_base + pred_resid

mae = mean_absolute_error(y_test_base, pred_final)
rmse = np.sqrt(mean_squared_error(y_test_base, pred_final))
r2 = r2_score(y_test_base, pred_final)
mape = mean_absolute_percentage_error(y_test_base, pred_final)

print("===== Week 6 最终模型性能 =====")
print(f"MAE: {mae:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f} | MAPE: {mape*100:.2f}%")

# ================= 4. 保存训练好的模型 =================
with open('Week6_Best_GBDT_Model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print("\n最佳模型已保存为 Week6_Best_GBDT_Model.pkl")

# ================= 5. SHAP 可解释性分析 =================
print("\n正在生成 SHAP 特征重要性图...")
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)

import matplotlib.pyplot as plt
# 生成 Summary Plot
plt.figure()
shap.summary_plot(shap_values, X_test, feature_names=feature_cols, show=False)
plt.savefig("Week6_SHAP_Summary.png", bbox_inches='tight', dpi=150)
print("SHAP 图已保存为 Week6_SHAP_Summary.png")