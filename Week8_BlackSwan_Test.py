"""Week 8 - Extreme black-swan stress test of the GBDT correction.

Question asked by the mentor:
    "Under an extreme black-swan scenario, can the GBDT model still give a
     reasonable correction margin?"

The script does four things:
  1. Builds a ladder of extreme scenarios (vol shock, spot crash, rate shock,
     combined black swan, historical COVID analogue).
  2. Runs BSM and the trained GBDT residual model on every scenario.
  3. Scores each output against explicit "reasonableness" rules.
  4. Writes a CSV plus two diagnostic charts.

Outputs
-------
Week8_BlackSwan_Results.csv
Week8_BlackSwan_Analysis.png
Week8_Correction_Saturation.png
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
import pricing_core as pc

K = config.K
T = config.T2
T1 = config.T1
REL_BOUND = 0.20  # a correction larger than 20% of the BSM price is flagged

model = pc.load_model()
train_min, train_max = pc.prediction_range(model)
quantiles = pc.feature_quantiles()


def run_case(group, name, S, K_, T_, r, vol, features):
    bsm = float(pc.bs_call(S, K_, T_, r, vol))
    corr = pc.predict_correction(model, features)
    final = bsm + corr
    rel = corr / bsm if bsm > 0 else np.nan
    guarded = pc.guard_correction(corr, bsm, REL_BOUND)
    ood = pc.ood_flags(features, quantiles)
    return {
        "Group": group,
        "Scenario": name,
        "S": S,
        "K": K_,
        "T": T_,
        "r": r,
        "Vol": vol,
        "BSM_Call": bsm,
        "GBDT_Correction": corr,
        "Final_Price": final,
        "Rel_Correction_Pct": rel * 100,
        "Guarded_Correction": guarded,
        "Guarded_Final": bsm + guarded,
        "Guarded_Rel_Pct": guarded / bsm * 100 if bsm > 0 else np.nan,
        "Direction_Positive": corr >= 0,
        "Magnitude_Within_20pct": abs(rel) <= REL_BOUND,
        "Within_Training_Range": train_min <= corr <= train_max,
        "Non_Negative": final >= 0,
        "OOD_Feature_Count": len(ood),
        "OOD_Features": ", ".join(ood),
        "Features": json.dumps(features),
    }


rows = []

# ---------------------------------------------------------------------------
# 1. Volatility shock ladder (spot at 150, rate 5%)
# ---------------------------------------------------------------------------
vol_ladder = [0.20, 0.30, 0.50, 0.80, 1.00, 1.20, 1.50]
for vol in vol_ladder:
    feats = pc.build_scenario_features(
        vol=vol, spot=150, rate=0.05, daily_return=0.0, vix_change=vol - 0.20,
        volume_mult=1.0 + 2.0 * (vol - 0.20), corr=-0.23 - 0.5 * (vol - 0.20)
    )
    rows.append(run_case("Volatility shock", f"Vol={vol:.0%}", 150, K, T, 0.05, vol, feats))

# ---------------------------------------------------------------------------
# 2. Spot crash ladder (vol fixed at 80%, rate cut to 1%)
# ---------------------------------------------------------------------------
spot_ladder = [150, 135, 120, 100, 80, 60]
for spot in spot_ladder:
    shock = (spot - 150) / 150
    feats = pc.build_scenario_features(
        vol=0.80, spot=spot, rate=0.01, daily_return=shock, vix_change=0.50,
        volume_mult=3.4, corr=-0.80
    )
    rows.append(run_case("Spot crash", f"S=${spot}", spot, K, T, 0.01, 0.80, feats))

# ---------------------------------------------------------------------------
# 3. Rate shock ladder (vol 50%, spot 150)
# ---------------------------------------------------------------------------
for rate in [0.05, 0.07, 0.09]:
    feats = pc.build_scenario_features(
        vol=0.50, spot=150, rate=rate, daily_return=0.0, vix_change=0.30,
        volume_mult=2.0, corr=-0.45
    )
    rows.append(run_case("Rate shock", f"r={rate:.0%}", 150, K, T, rate, 0.50, feats))

# ---------------------------------------------------------------------------
# 4. Combined black swan (the scenario from the original Week 8 draft)
# ---------------------------------------------------------------------------
rows.append(run_case(
    "Combined black swan", "S=100, vol=80%, r=1%",
    100, K, T, 0.01, 0.80, dict(pc.BLACK_SWAN_FEATURES)
))

# ---------------------------------------------------------------------------
# 5. Historical analogue: worst VIX day in the sample (COVID-19 crash)
# ---------------------------------------------------------------------------
df = pd.read_csv(pc.FEATURES_PATH)
df["Date"] = pd.to_datetime(df["Date"])
covid = df.loc[df["VIX"].idxmax()]
covid_feats = {name: float(covid[name]) for name in pc.FEATURE_ORDER}
covid_vol = float(covid["VIX"] / 100.0)
covid_rate = float(covid["DGS10"] / 100.0)
rows.append(run_case(
    "Historical analogue", f"COVID {covid['Date'].date()}",
    float(covid["Close"]), K, T, covid_rate, covid_vol, covid_feats
))

results = pd.DataFrame(rows)
results.to_csv("Week8_BlackSwan_Results.csv", index=False)

# ---------------------------------------------------------------------------
# Group-level diagnostics
# ---------------------------------------------------------------------------
vol_group = results[results["Group"] == "Volatility shock"]
spot_group = results[results["Group"] == "Spot crash"]

vol_corr_spread = vol_group["GBDT_Correction"].max() - vol_group["GBDT_Correction"].min()
spot_rel_min = spot_group["Rel_Correction_Pct"].min()
spot_rel_max = spot_group["Rel_Correction_Pct"].max()
pass_rate = results["Magnitude_Within_20pct"].mean()
direction_rate = results["Direction_Positive"].mean()
ood_rate = (results["OOD_Feature_Count"] > 0).mean()

print("=" * 72)
print("Week 8 - Extreme Black Swan Test of the GBDT Correction")
print("=" * 72)
print(f"Training output range of the GBDT : [{train_min:+.4f}, {train_max:+.4f}]")
print(f"Vol ladder correction spread       : {vol_corr_spread:.6f} USD")
print(f"Spot ladder relative correction    : {spot_rel_min:+.2f}% .. {spot_rel_max:+.2f}%")
print(f"Magnitude pass rate (|corr|<=20%)  : {pass_rate:.0%}")
print(f"Positive-direction rate            : {direction_rate:.0%}")
print(f"Cases with out-of-distribution x   : {ood_rate:.0%}")
print("-" * 72)
print(results[[
    "Scenario", "BSM_Call", "GBDT_Correction", "Final_Price",
    "Rel_Correction_Pct", "Magnitude_Within_20pct", "OOD_Feature_Count"
]].to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
print("-" * 72)

if vol_corr_spread < 1e-6:
    print("FINDING 1: The correction is completely flat across the whole volatility")
    print("           ladder. The tree model has saturated: it cannot price a")
    print("           volatility regime it has never seen in training.")
if spot_rel_max > 20:
    print("FINDING 2: For a deep crash the raw correction exceeds the 20% guardrail")
    print("           (the model keeps adding a fixed dollar amount to a call that")
    print("           is now far out-of-the-money).")
if results["Within_Training_Range"].all():
    print("FINDING 3: Every prediction stays inside the training output range,")
    print("           i.e. the model extrapolates by saturation, not by learning.")
print("VERDICT: Under an extreme black swan the GBDT residual does NOT provide a")
print("         economically reasonable correction margin. The tool therefore")
print("         applies a +/-20% guardrail and shows a warning to the user.")
print("=" * 72)

# ---------------------------------------------------------------------------
# Chart 1: 2x2 scenario dashboard
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax = axes[0, 0]
ax.plot(vol_group["Vol"] * 100, vol_group["BSM_Call"], "o-", label="BSM call", color="#1f77b4")
ax.plot(vol_group["Vol"] * 100, vol_group["Final_Price"], "s--", label="BSM + GBDT", color="#d62728")
ax.set_title("Volatility shock (S=150, r=5%)")
ax.set_xlabel("Implied volatility (%)")
ax.set_ylabel("Option price (USD)")
ax.legend()
ax.grid(alpha=0.3)

ax = axes[0, 1]
ax.plot(vol_group["Vol"] * 100, vol_group["GBDT_Correction"], "o-", color="#d62728",
        label="GBDT correction")
ax.axhspan(train_min, train_max, color="grey", alpha=0.20,
           label=f"Training output range [{train_min:.2f}, {train_max:.2f}]")
ax.set_title("Correction vs volatility: flat line = saturation")
ax.set_xlabel("Implied volatility (%)")
ax.set_ylabel("Correction (USD)")
ax.legend(loc="best")
ax.grid(alpha=0.3)

ax = axes[1, 0]
ax.plot(spot_group["S"], spot_group["BSM_Call"], "o-", label="BSM call", color="#1f77b4")
ax.plot(spot_group["S"], spot_group["Final_Price"], "s--", label="BSM + GBDT", color="#d62728")
ax.axhline(0, color="black", linewidth=0.8)
ax.set_title("Spot crash (vol=80%, r=1%)")
ax.set_xlabel("Spot price (USD)")
ax.set_ylabel("Option price (USD)")
ax.legend()
ax.grid(alpha=0.3)

ax = axes[1, 1]
ax.plot(spot_group["S"], spot_group["Rel_Correction_Pct"], "o-", color="#9467bd",
        label="Raw relative correction")
ax.axhspan(-REL_BOUND * 100, REL_BOUND * 100, color="green", alpha=0.12,
           label="+/-20% guardrail")
ax.axhline(0, color="black", linewidth=0.8)
ax.set_title("Relative correction vs spot")
ax.set_xlabel("Spot price (USD)")
ax.set_ylabel("Correction / BSM (%)")
ax.legend(loc="best")
ax.grid(alpha=0.3)

fig.suptitle("Week 8 - GBDT correction under extreme scenarios", fontsize=15)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("Week8_BlackSwan_Analysis.png", dpi=160)
plt.close(fig)

# ---------------------------------------------------------------------------
# Chart 2: saturation surface (correction as a function of vol and price gap)
# ---------------------------------------------------------------------------
vol_grid = np.linspace(0.10, 1.60, 31)
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

for spot, color in [(150, "#1f77b4"), (125, "#ff7f0e"), (100, "#d62728")]:
    corrs = []
    for vol in vol_grid:
        feats = pc.build_scenario_features(
            vol=float(vol), spot=spot, rate=0.05,
            daily_return=(spot - 150) / 150, vix_change=float(vol - 0.20),
            volume_mult=1.0 + 2.0 * (vol - 0.20), corr=-0.23 - 0.5 * (vol - 0.20)
        )
        corrs.append(pc.predict_correction(model, feats))
    axes[0].plot(vol_grid * 100, corrs, label=f"S=${spot}", color=color, linewidth=2)
axes[0].axhspan(train_min, train_max, color="grey", alpha=0.15)
axes[0].set_title("Correction vs volatility, by spot level")
axes[0].set_xlabel("Implied volatility (%)")
axes[0].set_ylabel("Correction (USD)")
axes[0].legend()
axes[0].grid(alpha=0.3)

gap_grid = np.linspace(-0.60, 0.10, 36)
corrs = []
for gap in gap_grid:
    spot = 150.0 * (1.0 + gap)
    feats = pc.build_scenario_features(
        vol=0.80, spot=spot, rate=0.01, daily_return=gap,
        vix_change=0.50, volume_mult=3.4, corr=-0.80
    )
    corrs.append(pc.predict_correction(model, feats))
axes[1].plot(gap_grid * 100, corrs, color="#d62728", linewidth=2)
axes[1].axhspan(train_min, train_max, color="grey", alpha=0.15,
                label=f"Training range [{train_min:.2f}, {train_max:.2f}]")
axes[1].axhline(0, color="black", linewidth=0.8)
axes[1].set_title("Correction vs distance from the 20D moving average")
axes[1].set_xlabel("MA_Gap (%)")
axes[1].set_ylabel("Correction (USD)")
axes[1].legend()
axes[1].grid(alpha=0.3)

fig.suptitle("Why the correction saturates: tree models cannot extrapolate", fontsize=15)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("Week8_Correction_Saturation.png", dpi=160)
plt.close(fig)

print("Saved: Week8_BlackSwan_Results.csv")
print("Saved: Week8_BlackSwan_Analysis.png")
print("Saved: Week8_Correction_Saturation.png")