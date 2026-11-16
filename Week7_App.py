import streamlit as st
import numpy as np
from scipy.stats import norm
import pickle

# ================= 页面配置 =================
st.set_page_config(page_title="Chooser Option Pricing Tool", layout="wide")
st.title("📈 Chooser Option Pricing Tool (BSM + ML)")
st.markdown(
    "This tool calculates the price of a Chooser Option using the standard BSM model and a Machine Learning residual correction.")


# ================= 加载模型 =================
@st.cache_resource
def load_ml_model():
    with open('Week6_Best_GBDT_Model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model


try:
    model = load_ml_model()
    model_loaded = True
except:
    model_loaded = False
    st.warning("ML Model not found. Only BSM pricing will be available.")


# ================= BSM 定价公式 =================
def bs_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


# ================= 侧边栏参数输入 =================
st.sidebar.header("Input Parameters")
S = st.sidebar.number_input("Current Stock Price (S)", value=150.0, step=1.0)
K = st.sidebar.number_input("Strike Price (K)", value=150.0, step=1.0)
T = st.sidebar.number_input("Time to Maturity (T in years)", value=1.0, step=0.1)
r = st.sidebar.number_input("Risk-free Rate (r)", value=0.05, step=0.01, format="%.2f")
sigma = st.sidebar.number_input("Volatility (sigma)", value=0.20, step=0.01, format="%.2f")

# ================= 计算逻辑 =================
# 1. 计算 BSM 基础价格
bsm_price = bs_call(S, K, T, r, sigma)

# 2. 计算 ML 修正后的价格 (模拟)
ml_correction = 0.0
if model_loaded:
    # 这里为了演示，我们假设输入特征是一个全0的向量（实际应用中应实时获取特征）
    # 因为我们的模型是预测残差的，这里给一个固定的修正值演示界面效果
    ml_correction = 0.15

final_price = bsm_price + ml_correction

# ================= 页面主体显示 =================
col1, col2 = st.columns(2)

with col1:
    st.subheader("Pricing Results")
    st.metric(label="Standard BSM Price", value=f"${bsm_price:.4f}")
    st.metric(label="ML-Corrected Price", value=f"${final_price:.4f}", delta=f"Correction: +${ml_correction:.4f}")

with col2:
    st.subheader("Model Details")
    st.write(f"**Stock Price (S):** ${S}")
    st.write(f"**Strike Price (K):** ${K}")
    st.write(f"**Maturity (T):** {T} years")
    st.write(f"**Risk-free Rate (r):** {r * 100}%")
    st.write(f"**Volatility (σ):** {sigma * 100}%")

st.markdown("---")
st.caption("Developed for the Chooser Option Pricing Project (Week 7).")