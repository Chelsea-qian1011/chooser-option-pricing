"""Generate supplementary figures used by the final report and the PPT."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({"font.size": 11})

# ---------------------------------------------------------------------------
# 1. Week 4 ablation: BSM price vs assumed volatility
# ---------------------------------------------------------------------------
abl = pd.read_csv("Week4_Ablation_Table.csv")
sig = abl["Volatility (Sigma)"].str.rstrip("%").astype(float)
fig, ax = plt.subplots(figsize=(7.2, 4.4))
ax.plot(sig, abl["BSM Price"], "o-", color="#1f77b4", label="BSM price")
ax.axhline(abl["Benchmark Price"].iloc[0], color="#d62728", linestyle="--",
           label=f"MC benchmark ${abl['Benchmark Price'].iloc[0]:.2f}")
ax.set_xlabel("Assumed constant volatility (%)")
ax.set_ylabel("Option price (USD)")
ax.set_title("Week 4 - BSM volatility ablation vs. Monte Carlo benchmark")
for x, y, e in zip(sig, abl["BSM Price"], abl["MAE"]):
    ax.annotate(f"MAE {e:.2f}", (x, y), textcoords="offset points",
                xytext=(0, 8), ha="center", fontsize=8)
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("Final_Report_Week4_Ablation.png", dpi=160)
plt.close(fig)

# ---------------------------------------------------------------------------
# 2. Week 5: predicted vs actual (Approach 2)
# ---------------------------------------------------------------------------
res = pd.read_csv("Week5_Approach2_Results.csv", parse_dates=["Date"])
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
axes[0].plot(res["Date"], res["Actual_Price"], label="Actual (proxy)", color="#111111", linewidth=1.4)
axes[0].plot(res["Date"], res["Pred_GBDT"], label="GBDT", color="#d62728", linewidth=1.2)
axes[0].plot(res["Date"], res["Pred_LR"], label="Linear", color="#1f77b4", linewidth=1.0, alpha=0.8)
axes[0].set_title("Week 5 - Approach 2: predicted vs. actual price")
axes[0].set_ylabel("Option price (USD)")
axes[0].tick_params(axis="x", rotation=30)
axes[0].legend()
axes[0].grid(alpha=0.25)

axes[1].scatter(res["Actual_Price"], res["Pred_GBDT"], s=18, color="#d62728", label="GBDT")
axes[1].scatter(res["Actual_Price"], res["Pred_LR"], s=14, color="#1f77b4", alpha=0.7, label="Linear")
lims = [res["Actual_Price"].min() - 1, res["Actual_Price"].max() + 1]
axes[1].plot(lims, lims, "k--", linewidth=1, label="45-degree line")
axes[1].set_xlabel("Actual price (USD)")
axes[1].set_ylabel("Predicted price (USD)")
axes[1].set_title("Prediction scatter")
axes[1].legend()
axes[1].grid(alpha=0.25)
fig.tight_layout()
fig.savefig("Final_Report_Week5_Performance.png", dpi=160)
plt.close(fig)

# ---------------------------------------------------------------------------
# 3. Project architecture
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(13, 3.4))
ax.set_xlim(0, 13)
ax.set_ylim(0, 3.4)
ax.axis("off")
boxes = [
    (0.2, "Data layer\nYahoo / FRED\nJPM, VIX, DGS10", "#dbeafe"),
    (2.7, "Feature layer\n13 engineered\nfeatures", "#e0f2fe"),
    (5.2, "Pricing layer\nRubinstein BSM\nchooser", "#fef3c7"),
    (7.7, "ML layer\nGBDT residual\ncorrection", "#fee2e2"),
    (10.2, "Delivery layer\nStreamlit tool\n+ reports", "#dcfce7"),
]
for x, text, color in boxes:
    box = FancyBboxPatch((x, 0.9), 2.0, 1.6, boxstyle="round,pad=0.08",
                         facecolor=color, edgecolor="#334155", linewidth=1.1)
    ax.add_patch(box)
    ax.text(x + 1.0, 1.7, text, ha="center", va="center", fontsize=10)
for x in [2.3, 4.8, 7.3, 9.8]:
    ax.annotate("", xy=(x + 0.4, 1.7), xytext=(x, 1.7),
                arrowprops=dict(arrowstyle="->", color="#475569", lw=1.4))
ax.text(6.5, 3.05, "Chooser option pricing pipeline", ha="center", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig("Final_Report_Architecture.png", dpi=160)
plt.close(fig)

# ---------------------------------------------------------------------------
# 4. Eight-week roadmap
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(13, 4.2))
ax.axis("off")
ax.set_xlim(0, 13.4)
ax.set_ylim(0, 4.0)
weeks = [
    ("Week 1", "Data sourcing\nAPI tests"),
    ("Week 2", "Cleaning &\nfeatures"),
    ("Week 3", "BSM chooser\nmodel"),
    ("Week 4", "Monte Carlo &\nbenchmark"),
    ("Week 5", "Two ML\narchitectures"),
    ("Week 6", "Tuning &\nSHAP"),
    ("Week 7", "Stress test &\nprototype"),
    ("Week 8", "Tool, black\nswan, delivery"),
]
x0, y0, w, h = 0.2, 1.0, 1.5, 1.5
for i, (week, text) in enumerate(weeks):
    x = x0 + i * 1.6
    color = "#dcfce7" if i == 7 else "#e2e8f0"
    box = FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0.05",
                         facecolor=color, edgecolor="#334155")
    ax.add_patch(box)
    ax.text(x + w / 2, y0 + h - 0.28, week, ha="center", fontsize=10, fontweight="bold")
    ax.text(x + w / 2, y0 + 0.55, text, ha="center", fontsize=9)
ax.plot([0.2, 13.2], [0.85, 0.85], color="#334155", linewidth=2)
ax.text(6.6, 3.55, "Eight-week project roadmap", ha="center", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig("Final_Report_Roadmap.png", dpi=160)
plt.close(fig)

print("Generated report figures.")