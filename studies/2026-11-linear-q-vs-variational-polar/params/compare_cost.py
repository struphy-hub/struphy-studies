"""Table of the cost runs (profile_cost.py): wall time per step and per unit of simulated time, per propagator, read
from cost_*.json beside this file (or $STUDY_RUNS); written to cost_table.md."""
import json
import os
from pathlib import Path

RUNS = Path(os.environ.get("STUDY_RUNS", Path(__file__).resolve().parent))
rows = []
for f in sorted(RUNS.glob("cost_*.json")):
    d = json.load(open(f))
    props = ", ".join(f"{k} {v:.2f}" for k, v in d["propagators"].items())
    rows.append(f"| {d['model']} | {d['dt']} | {d['s_per_step']:.2f} | {d['s_per_time']:.2f} | {props} |")
text = ("| model | dt | s per step | s per unit time | s per step and propagator |\n|---|---|---|---|---|\n"
        + "\n".join(rows) + "\n")
open(RUNS / "cost_table.md", "w").write(text)
print(text)
