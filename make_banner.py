import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

j = pd.read_csv("jpm_stock.csv", skiprows=[1], parse_dates=[0], index_col=0).iloc[1:]
j["Close"] = pd.to_numeric(j["Close"], errors="coerce")
j = j[j["Close"].notna()]
j.index = pd.to_datetime(j.index, errors="coerce")
j = j[j.index.notna()]

v = pd.read_csv("vix.csv", parse_dates=[0], index_col=0)["0"]
t = pd.read_csv("treasury.csv", parse_dates=[0], index_col=0)["0"]
v = pd.to_numeric(v, errors="coerce")
t = pd.to_numeric(t, errors="coerce")

bg = "#0E2A47"
fig, ax = plt.subplots(3, 1, figsize=(12, 4.15), sharex=True, facecolor=bg)
for a in ax:
    a.set_facecolor(bg)
    for s in a.spines.values():
        s.set_color("#3A5674")
        s.set_linewidth(0.6)
    a.tick_params(colors="#A8BED4", labelsize=9)
    a.grid(color="#223F5E", linewidth=0.5)
    a.set_axisbelow(True)

for a, s, c in zip(ax, [j["Close"], v, t], ["#C9A227", "#8FC7D8", "#D9C9A3"]):
    a.plot(s.index, s.values, color=c, linewidth=1.2)

fig.subplots_adjust(left=0.055, right=0.985, top=0.97, bottom=0.08, hspace=0.30)
fig.savefig("overview_banner.png", dpi=200, facecolor=bg)
print("saved")