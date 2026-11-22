"""Week 8 final deliverable - interactive Chooser Option pricing tool.

Run with:
    streamlit run Week8_App.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import config
import pricing_core as pc

st.set_page_config(
    page_title="Chooser Option Pricing Tool",
    page_icon="📈",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
REL_BOUND = 0.20

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    return pc.load_model(BASE_DIR / "Week6_Best_GBDT_Model.pkl")


@st.cache_data
def load_reference():
    quantiles = pc.feature_quantiles(BASE_DIR / "features.csv")
    model = load_model()
    train_min, train_max = pc.prediction_range(model, BASE_DIR / "features.csv")
    return quantiles, train_min, train_max


model = load_model()
quantiles, train_min, train_max = load_reference()

PRESETS = {
    "Normal market": pc.NORMAL_FEATURES,
    "High-volatility stress": pc.STRESS_FEATURES,
    "Black swan (COVID-style)": pc.BLACK_SWAN_FEATURES,
    "Custom": pc.NORMAL_FEATURES,
}

# ---------------------------------------------------------------------------
# Sidebar - market state
# ---------------------------------------------------------------------------
st.sidebar.title("Market state")

# The 10 GBDT features alone do not move the BSM price, so each preset also
# pins sigma. "Custom" leaves whatever the user has typed in place.
PRESET_SIGMA = {
    "Normal market": float(config.sigma),
    "High-volatility stress": 0.50,
    "Black swan (COVID-style)": 0.80,
    "Custom": None,
}

if "sigma" not in st.session_state:
    st.session_state["sigma"] = float(config.sigma)


def _apply_preset():
    target = PRESET_SIGMA.get(st.session_state["preset"])
    if target is not None:
        st.session_state["sigma"] = target


preset_name = st.sidebar.selectbox(
    "Scenario preset",
    list(PRESETS.keys()),
    index=0,
    key="preset",
    on_change=_apply_preset,
    help="Normal 20% / stress 50% / black swan 80% volatility.",
)

st.sidebar.subheader("Contract parameters")
S = st.sidebar.number_input("Spot price S ($)", min_value=1.0, value=150.0, step=1.0)
K = st.sidebar.number_input("Strike K ($)", min_value=1.0, value=float(config.K), step=1.0)
T1 = st.sidebar.number_input("Choice date T1 (years)", min_value=0.01, value=float(config.T1), step=0.05)
T2 = st.sidebar.number_input("Maturity T2 (years)", min_value=float(T1) + 0.01, value=float(config.T2), step=0.05)
r = st.sidebar.number_input("Risk-free rate r", min_value=0.0, max_value=0.30, value=float(config.r), step=0.005, format="%.3f")
sigma = st.sidebar.number_input("Volatility sigma", min_value=0.01, max_value=2.00, step=0.01, format="%.2f", key="sigma")

base_features = dict(PRESETS[preset_name])
linked = st.sidebar.checkbox(
    "Link market features to sigma and spot",
    value=True,
    help="Vol_5 / Vol_20 / Vol_60 and MA_Gap are derived from the contract inputs. "
         "Uncheck to enter all 10 GBDT features manually.",
)

if linked:
    features = pc.build_scenario_features(
        vol=sigma,
        spot=S,
        rate=r,
        daily_return=base_features["Return"],
        vix_change=base_features["VIX_Change"],
        volume_mult=base_features["Volume_MA5"] / pc.NORMAL_FEATURES["Volume_MA5"],
        corr=base_features["VIX_JPM_Corr"],
        base=base_features,
    )
else:
    with st.sidebar.expander("GBDT features (manual)", expanded=True):
        features = {}
        for name in pc.FEATURE_ORDER:
            features[name] = st.number_input(
                f"{pc.FEATURE_LABELS[name]} [{name}]",
                value=float(base_features[name]),
                step=0.01,
                format="%.4f",
                key=f"feat_{name}",
            )

# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------
result = pc.price_chooser(
    S, K, T1, T2, r, sigma, model=model, features=features, max_rel=REL_BOUND
)
ood = pc.ood_flags(features, quantiles)
guarded = abs(result["ml_correction_raw"] - result["ml_correction"]) > 1e-12

st.title("Chooser Option Pricing Tool")
st.caption(
    "Dual pricing: Rubinstein BSM chooser vs. BSM + GBDT residual correction. "
    "The correction is capped at +/-20% of the BSM price to stay economically sensible."
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("BSM chooser price", f"${result['bsm_price']:,.2f}")
m2.metric("GBDT-corrected price", f"${result['final_price']:,.2f}")
m3.metric("Correction margin", f"${result['ml_correction']:+,.4f}")
m4.metric("Margin / BSM", f"{result['error_pct'] * 100:,.3f}%")

if guarded:
    st.warning(
        f"Raw GBDT correction ${result['ml_correction_raw']:+,.2f} exceeded the "
        f"+/-{REL_BOUND:.0%} guardrail and was capped. This usually means the "
        "scenario sits outside the training sample."
    )
if ood:
    st.info("Out-of-distribution feature(s): " + ", ".join(ood))

tabs = st.tabs(["Pricing", "Sensitivity", "Model diagnostics", "About"])

# ---------------------------------------------------------------------------
# Tab 1 - pricing
# ---------------------------------------------------------------------------
with tabs[0]:
    left, right = st.columns([1, 1])

    with left:
        st.subheader("Price comparison")
        chart_df = pd.DataFrame(
            {"Price (USD)": [result["bsm_price"], result["final_price"]]},
            index=["BSM chooser", "BSM + GBDT"],
        )
        st.bar_chart(chart_df, horizontal=True, height=320, color="#1f77b4")

        st.subheader("Error margin display")
        st.write(
            f"The GBDT model shifts the BSM price by **${result['ml_correction']:+,.4f}** "
            f"(**{result['error_pct'] * 100:,.3f}%** of the BSM value)."
        )
        band = REL_BOUND * result["bsm_price"]
        st.markdown(
            f"Guardrail band: **±\\${band:,.2f}**. "
            f"Raw model output before the guardrail: "
            f"**\\${result['ml_correction_raw']:+,.4f}**."
        )

    with right:
        st.subheader("Contract breakdown")
        breakdown = pd.DataFrame(
            {
                "Value": [
                    f"${S:,.2f}",
                    f"${K:,.2f}",
                    f"{T1:.2f} years",
                    f"{T2:.2f} years",
                    f"{r:.2%}",
                    f"{sigma:.1%}",
                    f"${result['call_price']:,.4f}",
                    f"${result['bsm_price'] - result['call_price']:,.4f}",
                ]
            },
            index=[
                "Spot S",
                "Strike K",
                "Choice date T1",
                "Maturity T2",
                "Risk-free rate r",
                "Volatility sigma",
                "Call leg (maturity T2)",
                "Put leg (maturity T1)",
            ],
        )
        st.dataframe(breakdown, height=330, width="stretch")

        out = pd.DataFrame(
            [{
                "Preset": preset_name,
                "S": S, "K": K, "T1": T1, "T2": T2, "r": r, "sigma": sigma,
                **features,
                "BSM_Price": result["bsm_price"],
                "GBDT_Correction_Raw": result["ml_correction_raw"],
                "GBDT_Correction_Guarded": result["ml_correction"],
                "Final_Price": result["final_price"],
            }]
        )
        st.download_button(
            "Download scenario result (CSV)",
            out.to_csv(index=False).encode("utf-8"),
            file_name="chooser_pricing_scenario.csv",
            mime="text/csv",
        )

# ---------------------------------------------------------------------------
# Tab 2 - sensitivity
# ---------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Price sensitivity")

    col_a, col_b = st.columns(2)

    with col_a:
        vol_grid = np.linspace(0.10, 1.50, 29)
        vol_rows = []
        for vol in vol_grid:
            f = pc.build_scenario_features(
                vol=float(vol), spot=S, rate=r,
                daily_return=features["Return"], vix_change=features["VIX_Change"],
                volume_mult=features["Volume_MA5"] / pc.NORMAL_FEATURES["Volume_MA5"],
                corr=features["VIX_JPM_Corr"],
            )
            bsm = pc.chooser_option_price(S, K, T1, T2, r, float(vol))
            corr = pc.guard_correction(pc.predict_correction(model, f), bsm, REL_BOUND)
            vol_rows.append({"Volatility": float(vol), "BSM": bsm, "BSM + GBDT": bsm + corr})
        vol_df = pd.DataFrame(vol_rows).set_index("Volatility")
        st.caption("Price vs volatility")
        st.line_chart(vol_df, height=300)
        st.caption("The corrected price tracks BSM closely because the GBDT correction is a small dollar spread, not a volatility model.")

    with col_b:
        spot_grid = np.linspace(max(K * 0.35, 1.0), K * 1.5, 29)
        spot_rows = []
        for spot in spot_grid:
            f = pc.build_scenario_features(
                vol=sigma, spot=float(spot), rate=r,
                daily_return=features["Return"], vix_change=features["VIX_Change"],
                volume_mult=features["Volume_MA5"] / pc.NORMAL_FEATURES["Volume_MA5"],
                corr=features["VIX_JPM_Corr"],
            )
            bsm = pc.chooser_option_price(float(spot), K, T1, T2, r, sigma)
            corr = pc.guard_correction(pc.predict_correction(model, f), bsm, REL_BOUND)
            spot_rows.append({"Spot": float(spot), "BSM": bsm, "BSM + GBDT": bsm + corr})
        spot_df = pd.DataFrame(spot_rows).set_index("Spot")
        st.caption("Price vs spot")
        st.line_chart(spot_df, height=300)
        st.caption("Deep out-of-the-money scenarios trigger the guardrail, keeping the correction bounded.")

    st.subheader("Correction margin across volatility")
    corr_rows = []
    for vol in np.linspace(0.10, 1.50, 29):
        f = pc.build_scenario_features(
            vol=float(vol), spot=S, rate=r,
            daily_return=features["Return"], vix_change=features["VIX_Change"],
            volume_mult=features["Volume_MA5"] / pc.NORMAL_FEATURES["Volume_MA5"],
            corr=features["VIX_JPM_Corr"],
        )
        bsm = pc.chooser_option_price(S, K, T1, T2, r, float(vol))
        raw = pc.predict_correction(model, f)
        corr_rows.append({
            "Volatility": float(vol),
            "Raw correction": raw,
            "Guarded correction": pc.guard_correction(raw, bsm, REL_BOUND),
        })
    st.line_chart(pd.DataFrame(corr_rows).set_index("Volatility"), height=280)
    st.caption(
        "A flat raw line means the tree model has saturated; it is not a "
        "volatility risk-premium model. The guarded line is what the tool uses."
    )

# ---------------------------------------------------------------------------
# Tab 3 - diagnostics
# ---------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Model diagnostics")
    d1, d2, d3 = st.columns(3)
    d1.metric("Model", "GradientBoostingRegressor")
    d2.metric("Training output range", f"${train_min:+.2f} .. ${train_max:+.2f}")
    d3.metric("OOD features", str(len(ood)))

    st.write(
        "The GBDT is trained on 10 daily market features to predict a **residual "
        "correction** on top of the BSM call price. Because a tree ensemble can "
        "only output values between the leaves it has seen, extreme scenarios "
        "saturate at the training boundary."
    )

    st.subheader("Current features vs. training distribution")
    current = pd.DataFrame({
        "Current": [features[n] for n in pc.FEATURE_ORDER],
        "Training p1": [quantiles.loc[0.01, n] for n in pc.FEATURE_ORDER],
        "Training median": [quantiles.loc[0.50, n] for n in pc.FEATURE_ORDER],
        "Training p99": [quantiles.loc[0.99, n] for n in pc.FEATURE_ORDER],
    }, index=pc.FEATURE_ORDER)
    current["Status"] = [
        "OUT OF RANGE" if n in ood else "inside" for n in pc.FEATURE_ORDER
    ]
    st.dataframe(
        current.style.format({
            "Current": "{:,.4f}",
            "Training p1": "{:,.4f}",
            "Training median": "{:,.4f}",
            "Training p99": "{:,.4f}",
        }),
        width="stretch",
    )

    if ood:
        st.error(
            "This scenario is outside the 1%-99% training range for: "
            + ", ".join(ood)
            + ". Treat the raw GBDT output with caution; the guardrail is active."
        )
    else:
        st.success("All features are inside the 1%-99% training range.")

    with st.expander("Model hyper-parameters"):
        st.json(model.get_params())

# ---------------------------------------------------------------------------
# Tab 4 - about
# ---------------------------------------------------------------------------
with tabs[3]:
    st.subheader("Methodology")
    st.markdown(
        """
**Chooser option (Rubinstein, 1991).** At the choice date `T1` the holder
decides whether the contract is a call or a put with strike `K` maturing at
`T2`. The payoff is replicated by:

* a call with strike `K` and maturity `T2`, plus
* a put with strike `K * exp(-r * (T2 - T1))` and maturity `T1`.

**Machine-learning correction.** A gradient-boosted tree model is trained on
daily JPM features (returns, realised volatility, volume, VIX changes and
correlation, rate momentum, moving-average gap) to predict the residual
between the BSM price and its next-day value. The prediction is added to the
BSM chooser price as a dollar spread.

**Guardrail.** Tree models cannot extrapolate. Under extreme scenarios the
raw correction saturates at the training range (about -5.37 to +3.83 USD),
which can be 80%+ of a deep out-of-the-money option. The tool therefore caps
the correction at +/-20% of the BSM price and warns when the cap is hit.

**Known limitation.** The project does not have exchange-traded option prices;
the target is a VIX-implied BSM proxy. The model should be read as a
demonstration of the hybrid workflow, not as a production pricing engine.
"""
    )
    st.caption("Week 8 final tool - Chooser Option Pricing with Real-World Data & Machine Learning.")