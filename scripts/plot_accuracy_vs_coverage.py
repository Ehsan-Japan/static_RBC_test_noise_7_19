"""
plot_accuracy_vs_coverage.py
────────────────────────────
Reads evaluation.txt from the first 100 samples and plots
Accuracy vs. Voltage Grid Coverage.

Accuracy definition (from evaluator.py):
    accuracy = (TP + TN) / total_pixels * 100
    where total_pixels = x_resolution * y_resolution = 10,000
    TP = sweep-detected pixels that are true transition-line pixels
    TN = non-detected pixels that are truly background

NOTE: because transition lines are sparse (~3–5% of the grid), TN dominates
and accuracy is high even for imperfect detectors. Recall is a stricter
measure of how many true transitions were found, and is printed alongside.

Usage:
    python scripts/plot_accuracy_vs_coverage.py
"""

import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..",
    "training_data",
    "num_1800_rays_6_res_100_image_res_100",
)
N_SAMPLES  = 100
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

# ── Parsing ───────────────────────────────────────────────────────────────────

def _float(pattern: str, text: str) -> float:
    m = re.search(pattern, text)
    if m is None:
        raise ValueError(f"Pattern not found: {pattern!r}")
    return float(m.group(1))

def parse_evaluation(path: str) -> dict:
    with open(path) as f:
        txt = f.read()
    return {
        "accuracy": _float(r"Accuracy\s*:\s*([\d.]+)%",      txt),
        "recall":   _float(r"Recall\s*:\s*([\d.]+)%",        txt),
        "coverage": _float(r"Grid Coverage\s*:\s*([\d.]+)%", txt),
    }

# ── Data collection ───────────────────────────────────────────────────────────

records = []
missing = []

for i in range(1, N_SAMPLES + 1):
    path = os.path.join(DATA_DIR, f"sample_{i}", "evaluation.txt")
    if not os.path.isfile(path):
        missing.append(i)
        continue
    try:
        records.append(parse_evaluation(path))
    except ValueError as e:
        print(f"[WARN] sample_{i}: {e}")

print(f"Loaded {len(records)} / {N_SAMPLES} samples"
      + (f"  (missing: {missing})" if missing else ""))

if not records:
    raise SystemExit("No data found — check DATA_DIR.")

coverage = np.array([r["coverage"] for r in records])
accuracy = np.array([r["accuracy"] for r in records])
recall   = np.array([r["recall"]   for r in records])

# ── Statistics ────────────────────────────────────────────────────────────────

print(f"\n── Summary (N={len(records)}) ──")
for name, arr in [("Coverage (%)", coverage), ("Accuracy (%)", accuracy), ("Recall  (%)", recall)]:
    print(f"  {name}  mean={arr.mean():.2f}  std={arr.std():.2f}"
          f"  min={arr.min():.2f}  max={arr.max():.2f}")

# ── Plot: Accuracy vs Coverage ────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

fig, ax = plt.subplots(figsize=(7, 5))

ax.scatter(coverage, accuracy, color="steelblue", s=30, alpha=0.75,
           edgecolors="white", linewidths=0.4, label="Samples (N=100)")

# Linear trend
z = np.polyfit(coverage, accuracy, 1)
x_fit = np.linspace(coverage.min(), coverage.max(), 300)
ax.plot(x_fit, np.polyval(z, x_fit), color="steelblue", linewidth=2,
        linestyle="--", label=f"Linear fit  (slope = {z[0]:+.3f} %/% cov.)")

# Mean reference lines
ax.axhline(accuracy.mean(), color="gray", linewidth=1, linestyle=":",
           label=f"Mean accuracy = {accuracy.mean():.1f}%")
ax.axvline(coverage.mean(), color="lightgray", linewidth=1, linestyle=":")

ax.set_xlabel("Voltage Grid Coverage (%)", fontsize=13)
ax.set_ylabel("Pixel-Level Accuracy (%)\n(TP + TN) / 10 000", fontsize=13)
ax.set_title(
    "Accuracy vs. Voltage Grid Coverage\n"
    "DQD Transition-Line Detection  |  6 rays × 100 pts  |  100×100 grid",
    fontsize=11,
)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

out = os.path.join(OUTPUT_DIR, "accuracy_vs_coverage.png")
plt.tight_layout()
plt.savefig(out, dpi=300, bbox_inches="tight")
print(f"\nSaved → {out}")
