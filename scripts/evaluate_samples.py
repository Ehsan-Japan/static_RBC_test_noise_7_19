"""
evaluate_samples.py
───────────────────
Reads evaluation.txt from the first 100 samples and prints a full
statistical summary of all detection metrics and grid coverage.

Metrics reported
────────────────
  Accuracy  = (TP + TN) / total_pixels
              Pixel-level correctness over the full 100×100 grid.
              NOTE: dominated by true-negative (background) pixels
              because transition lines are sparse (~3–5% of the grid).

  Recall    = TP / (TP + FN)
              Fraction of true transition-line pixels that were found.
              This is the primary measure of detection completeness.

  Precision = TP / (TP + FP)
              Fraction of detected pixels that are true transitions.

  F1 Score  = 2 · Precision · Recall / (Precision + Recall)
              Harmonic mean of precision and recall.

  Coverage  = scanned_pixels / total_pixels
              Fraction of the voltage grid actually measured.

Usage:
    python scripts/evaluate_samples.py
"""

import os
import re

DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..",
    "training_data",
    "num_30_rays_6_res_100_image_res_100",
)
N_SAMPLES = 30


# ── Parsing ───────────────────────────────────────────────────────────────────

def _float(pattern, text):
    m = re.search(pattern, text)
    if m is None:
        raise ValueError(f"Pattern not found: {pattern!r}")
    return float(m.group(1))

def _int(pattern, text):
    m = re.search(pattern, text)
    if m is None:
        raise ValueError(f"Pattern not found: {pattern!r}")
    return int(m.group(1))

def parse_evaluation(path):
    with open(path) as f:
        txt = f.read()
    return {
        "gt_peaks":   _int(  r"Ground Truth Peaks\s*:\s*(\d+)",    txt),
        "det_peaks":  _int(  r"Detected Peaks\s*:\s*(\d+)",         txt),
        "tp":         _int(  r"True Positives\s*:\s*(\d+)",         txt),
        "fp":         _int(  r"False Positives\s*:\s*(\d+)",        txt),
        "fn":         _int(  r"False Negatives\s*:\s*(\d+)",        txt),
        "precision":  _float(r"Precision\s*:\s*([\d.]+)%",          txt),
        "recall":     _float(r"Recall\s*:\s*([\d.]+)%",             txt),
        "f1":         _float(r"F1 Score\s*:\s*([\d.]+)%",           txt),
        "accuracy":   _float(r"Accuracy\s*:\s*([\d.]+)%",           txt),
        "scanned":    _int(  r"Scanned Cells\s*:\s*(\d+)",          txt),
        "total":      _int(  r"Scanned Cells\s*:\s*\d+\s*\(out of\s*(\d+)\)", txt),
        "coverage":   _float(r"Grid Coverage\s*:\s*([\d.]+)%",      txt),
        "num_rays":   _int(  r"Number of Rays\s*:\s*(\d+)",         txt),
        "ray_res":    _int(  r"Ray Resolution\s*:\s*(\d+)",         txt),
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

N = len(records)

if N == 0:
    raise SystemExit("No evaluation files found. Check DATA_DIR.")

if missing:
    print(f"[INFO] Missing samples: {missing}\n")


# ── Helpers ───────────────────────────────────────────────────────────────────

def col(key):
    """Extract a list of values for the given key across all records."""
    return [r[key] for r in records]

def stats(values, label, unit="%", fmt=".2f"):
    vals = values
    mean = sum(vals) / len(vals)
    variance = sum((v - mean) ** 2 for v in vals) / len(vals)
    std  = variance ** 0.5
    print(f"  {label:<20s}  mean={mean:{fmt}}{unit}  "
          f"std={std:{fmt}}{unit}  "
          f"min={min(vals):{fmt}}{unit}  "
          f"max={max(vals):{fmt}}{unit}")
    return mean, std, min(vals), max(vals)


# ── Print report ──────────────────────────────────────────────────────────────

SEP  = "=" * 65
sep  = "-" * 65

print()
print(SEP)
print("  DQD Transition-Line Detection — Evaluation Summary")
print(SEP)
print(f"  Samples evaluated  : {N}")
print(f"  Rays per sample    : {records[0]['num_rays']}")
print(f"  Points per ray     : {records[0]['ray_res']}")
print(f"  Grid size          : {records[0]['total']} pixels  "
      f"({int(records[0]['total']**0.5)}×{int(records[0]['total']**0.5)})")
print()

# --- Pixel counts ---
print("Peak / Pixel Counts (per sample)")
print(sep)
stats(col("gt_peaks"),  "Ground truth peaks", unit=" px")
stats(col("det_peaks"), "Detected peaks",     unit=" px")
stats(col("tp"),        "True positives",     unit=" px")
stats(col("fp"),        "False positives",    unit=" px")
stats(col("fn"),        "False negatives",    unit=" px")
print()

# --- Detection metrics ---
print("Detection Metrics")
print(sep)
print()
print("  Accuracy  = (TP + TN) / total_pixels")
print("  ► High by construction because TN (background) dominates")
print("    the sparse transition-line grid (~3–5% foreground).")
stats(col("accuracy"),  "Accuracy")
print()
print("  Recall    = TP / (TP + FN)")
print("  ► Primary quality metric: fraction of true transitions found.")
stats(col("recall"),    "Recall")
print()
print("  Precision = TP / (TP + FP)")
print("  ► Low because sweeps generate many candidate points.")
stats(col("precision"), "Precision")
print()
print("  F1 Score  = 2·P·R / (P+R)   (harmonic mean)")
stats(col("f1"),        "F1 Score")
print()

# --- Coverage ---
print("Measurement Efficiency")
print(sep)
stats(col("scanned"), "Scanned cells", unit=" px", fmt=".1f")
stats(col("coverage"), "Grid coverage")
print()

# --- Efficiency ratio ---
mean_recall   = sum(col("recall"))   / N
mean_coverage = sum(col("coverage")) / N
ratio = mean_recall / mean_coverage
print(f"  Recall / Coverage ratio  :  {ratio:.2f}x")
print(f"  → On average the method finds {mean_recall:.1f}% of transitions")
print(f"    while scanning only {mean_coverage:.1f}% of the grid.")
print()
print(SEP)
