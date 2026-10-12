import pandas as pd
import numpy as np

# ===== 1. 读取 JPM（只保留 Date, Close, Volume） =====
jpm = pd.read_csv("jpm_stock.csv", skiprows=[1, 2])
jpm.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]
jpm = jpm[["Date", "Close", "Volume"]]          # 删掉 High / Low / Open
jpm["Date"] = pd.to_datetime(jpm["Date"])
jpm = jpm.sort_values("Date").reset_index(drop=True)

# ===== 2. 读取 Treasury =====
treasury = pd.read_csv("treasury.csv", index_col=0, names=["Date", "DGS10"])
treasury.index = pd.to_datetime(treasury.index)
treasury = treasury.sort_index()
treasury["DGS10"] = treasury["DGS10"].ffill()    # 只用 ffill

# ===== 3. 读取 VIX =====
vix = pd.read_csv("vix.csv", index_col=0, names=["Date", "VIX"])
vix.index = pd.to_datetime(vix.index)
vix = vix.sort_index()
vix["VIX"] = vix["VIX"].ffill()                  # 只用 ffill

# ===== 4. 合并 =====
treasury = treasury.reset_index().rename(columns={"index": "Date"})
vix = vix.reset_index().rename(columns={"index": "Date"})
df = jpm.merge(treasury, on="Date", how="left")
df = df.merge(vix, on="Date", how="left")
df["DGS10"] = df["DGS10"].ffill()
df["VIX"] = df["VIX"].ffill()

# ===== 5. 特征工程 =====
df["Return"] = df["Close"].pct_change()
df["Vol_5"] = df["Return"].rolling(5).std() * np.sqrt(252)
df["Vol_20"] = df["Return"].rolling(20).std() * np.sqrt(252)
df["Vol_60"] = df["Return"].rolling(60).std() * np.sqrt(252)
df["Volume_MA5"] = df["Volume"].rolling(5).mean()
df["Volume_Change"] = df["Volume"].pct_change()
df["VIX_Change"] = df["VIX"].pct_change()
df["VIX_JPM_Corr"] = df["VIX"].rolling(20).corr(df["Return"])
df["Rate_Momentum"] = df["DGS10"].diff()
df["MA5"] = df["Close"].rolling(5).mean()
df["MA20"] = df["Close"].rolling(20).mean()

# 新增两个特征
df["Return_Lag1"] = df["Return"].shift(1)
df["MA_Gap"] = (df["MA5"] - df["MA20"]) / df["Close"]

# ===== 6. 删除空值 =====
df = df.dropna().reset_index(drop=True)

# ===== 7. 保存 =====
df.to_csv("features.csv", index=False)
df.to_parquet("features.parquet", index=False)

print("完成！")
print("行数:", len(df))
print("列数:", len(df.columns))
print("列名:", df.columns.tolist())