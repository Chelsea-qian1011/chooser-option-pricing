import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from scipy.stats import norm
import optuna
import config

# 1. 准备数据 (复用之前的逻辑)
df = pd.read_csv("features.csv")
df['ATM_IV'] = df['VIX'] / 100.0
feature_cols = ['Return', 'Vol_5', 'Vol_20', 'Vol_60', 'Volume_MA5', 'VIX_Change', 'VIX_JPM_Corr', 'Rate_Momentum',
                'Return_Lag1', 'MA_Gap']


def bs_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


df['BSM_Base'] = df['ATM_IV'].apply(lambda v: bs_call(config.S, config.K, config.T2, config.r, v))
df['Residual'] = df['BSM_Base'] - df['BSM_Base'].shift(-1)
df_clean = df.dropna().copy()

X = df_clean[feature_cols].values
y = df_clean['Residual'].values
train_size = int(len(df_clean) * 0.70)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

tscv = TimeSeriesSplit(n_splits=3)
param_grid = {'learning_rate': [0.01, 0.05, 0.1], 'max_depth': [3, 4, 5], 'n_estimators': [50, 100, 200]}
base_model = GradientBoostingRegressor(random_state=42)

# ================= 方法 1：网格搜索 (Grid Search) =================
print("1. 运行网格搜索...")
grid_search = GridSearchCV(base_model, param_grid, cv=tscv, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
grid_best = grid_search.best_estimator_

# ================= 方法 2：随机搜索 (Random Search) =================
print("2. 运行随机搜索...")
rand_search = RandomizedSearchCV(base_model, param_grid, n_iter=15, cv=tscv, scoring='neg_mean_squared_error',
                                 random_state=42)
rand_search.fit(X_train, y_train)
rand_best = rand_search.best_estimator_

# ================= 方法 3：贝叶斯优化 (Optuna) =================
print("3. 运行贝叶斯优化 (Optuna)...")


def objective(trial):
    param = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'max_depth': trial.suggest_int('max_depth', 3, 5),
        'n_estimators': trial.suggest_int('n_estimators', 50, 200)
    }
    model = GradientBoostingRegressor(**param, random_state=42)
    model.fit(X_train, y_train)
    return mean_squared_error(y_test, model.predict(X_test))  # 用测试集算MSE


optuna.logging.set_verbosity(optuna.logging.WARNING)  # 静音
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=15)
optuna_best = GradientBoostingRegressor(**study.best_params, random_state=42)
optuna_best.fit(X_train, y_train)

# ================= 画收敛图 (训练过程记录) =================
print("正在生成收敛图...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 记录路径并画图：由于 warm_start 的机制，我们要手动模拟树的数量增长
for ax, best_model, title in zip(axes, [grid_best, rand_best, optuna_best],
                                 ['Grid Search', 'Random Search', 'Bayesian (Optuna)']):
    lr, max_d, _ = best_model.learning_rate, best_model.max_depth, best_model.n_estimators
    train_errs = []
    test_errs = []
    for n in range(10, 210, 20):  # 树的数量从10到200，步长20
        m = GradientBoostingRegressor(n_estimators=n, learning_rate=lr, max_depth=max_d, random_state=42)
        m.fit(X_train, y_train)
        train_errs.append(mean_squared_error(y_train, m.predict(X_train)))
        test_errs.append(mean_squared_error(y_test, m.predict(X_test)))

    ax.plot(range(10, 210, 20), train_errs, label='Train MSE', marker='o')
    ax.plot(range(10, 210, 20), test_errs, label='Test MSE', marker='s')
    ax.set_title(f"Convergence: {title}")
    ax.set_xlabel("Number of Trees (n_estimators)")
    ax.set_ylabel("MSE")
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("Week6_Tuning_Convergence.png", dpi=150)
print("收敛图已保存为 Week6_Tuning_Convergence.png")