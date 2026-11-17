"""Energy behaviour of the time runs (params_linear.py): relative change of the scalar en_tot of each run against time,
written to compare_energy.png and compare_energy.md. Run folders are read beside this file (or from $STUDY_RUNS):
q_stable_dt0.5, var_stable_dt0.5, q_stable_dt0.125, var_stable_dt0.125 (missing ones are skipped)."""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from struphy import Output

HERE = Path(__file__).resolve().parent
RUNS = Path(os.environ.get("STUDY_RUNS", HERE))
CASES = [("q", "0.5"), ("var", "0.5"), ("q", "0.125"), ("var", "0.125")]
COLORS = {"q": "#2a78d6", "var": "#eb6834"}
STYLES = {"0.5": "-", "0.125": (0, (4, 3))}

plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": "#52514e",
                     "axes.grid": True, "grid.color": "#e4e3df", "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 11})
fig, ax = plt.subplots(figsize=(8, 4.2))
rows = []
for model, dt in CASES:
    folder = RUNS / f"{model}_stable_dt{dt}"
    if not (folder / "run_metadata.json").exists():
        continue
    sc = Output(str(folder)).scalars
    t = np.asarray(sc["t"])
    e = np.asarray(sc["en_tot"]).ravel()
    rel = (e - e[0]) / abs(e[0])
    ax.semilogy(t, np.abs(rel) + 1e-17, color=COLORS[model], ls=STYLES[dt], lw=2,
                label=f"{'q model' if model == 'q' else 'variational'}, dt = {dt}")
    rows.append((model, dt, float(np.max(np.abs(rel))), float(rel[-1])))
ax.set_xlabel("t")
ax.set_ylabel("|en_tot(t) - en_tot(0)| / en_tot(0)")
ax.set_title("Stable screw pinch, polar cylinder 16 x 12 x 12: change of en_tot", loc="left", fontsize=11)
ax.legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(RUNS / "compare_energy.png", dpi=150)
with open(RUNS / "compare_energy.md", "w") as f:
    f.write("| model | dt | max relative change of en_tot | at the end |\n|---|---|---|---|\n")
    for model, dt, mx, end in rows:
        f.write(f"| {model} | {dt} | {mx:.2e} | {end:+.2e} |\n")
print(open(RUNS / "compare_energy.md").read())
