import numpy as np
from scipy.stats import norm

def bs_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def bs_put(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def chooser_option_price(S, K, T1, T2, r, sigma):
    call_price = bs_call(S, K, T2, r, sigma)
    K_put = K * np.exp(-r * (T2 - T1))
    put_price = bs_put(S, K_put, T1, r, sigma)
    return call_price + put_price

# 测试验证
S = 150; K = 150; T1 = 0.5; T2 = 1.0; r = 0.05; sigma = 0.20
price = chooser_option_price(S, K, T1, T2, r, sigma)

print(f"BSM 计算出的 Chooser Option 价格: ${price:.4f}")
import numpy as np
from scipy.stats import norm

def bs_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def bs_put(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def chooser_option_price(S, K, T1, T2, r, sigma):
    call_price = bs_call(S, K, T2, r, sigma)
    K_put = K * np.exp(-r * (T2 - T1))
    put_price = bs_put(S, K_put, T1, r, sigma)
    return call_price + put_price

# 测试验证
S = 150; K = 150; T1 = 0.5; T2 = 1.0; r = 0.05; sigma = 0.20
price = chooser_option_price(S, K, T1, T2, r, sigma)

print(f"BSM 计算出的 Chooser Option 价格: ${price:.4f}")