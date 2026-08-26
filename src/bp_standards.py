"""
Blood-pressure measurement-device validation metrics: BHS, AAMI (SP10),
ISO 81060-2:2018, and IEEE 1708.

These are the standards used to grade a cuff/cuffless BP device against a
reference. They all operate on the error distribution

    error = estimate - reference          (mmHg, per paired reading)

computed separately for SBP and DBP. Nothing here fits or trains anything -
you hand it the predicted vs. true BP (absolute mmHg) for every test reading
and it returns the grades.

IMPORTANT context for THIS project (SKNA -> BP):
  * These standards judge ABSOLUTE BP error. Our model is calibrated (predicts
    delta-from-calibration) and evaluated LOSO, so the pooled cross-subject
    error still includes the per-subject calibration offset the model cannot
    infer (see the calibration audit in diagnostics.py). The grades below are
    therefore an honest device-level verdict, not a measure of tracking skill.
  * Per the project's own finding, the model does not beat the zero-delta
    baseline, so expect Grade D / FAIL. Report these alongside the zero-delta
    baseline MAE, never on their own - a passing-looking number here would be
    the calibration reading doing the work, not the model.
  * BHS/AAMI/ISO were written for >=85 SUBJECTS with several readings each.
    With a handful of subjects the pass/fail is indicative only; the functions
    still compute it but also report n and n_subjects so you can caveat it.

Apply on the POOLED predictions across all LOSO folds (one row per test
reading, all held-out subjects concatenated) - a single fold is one subject
and far too small for any of these.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# BHS - British Hypertension Society protocol
# --------------------------------------------------------------------------
# Grade by the cumulative percentage of readings whose ABSOLUTE error falls
# within 5 / 10 / 15 mmHg. A grade is awarded only if ALL THREE thresholds are
# met; the assigned grade is the best (A>B>C) that qualifies, else D.
#
#   Grade   <=5 mmHg   <=10 mmHg   <=15 mmHg
#     A        60%        85%         95%
#     B        50%        75%         90%
#     C        40%        65%         85%
#     D                 worse than C
BHS_GRADES = (
    ("A", 60.0, 85.0, 95.0),
    ("B", 50.0, 75.0, 90.0),
    ("C", 40.0, 65.0, 85.0),
)


def bhs_grade(errors):
    """errors: 1-D array of (estimate - reference). Returns a dict."""
    abs_err = np.abs(np.asarray(errors, dtype=np.float64))
    n = abs_err.size
    if n == 0:
        return {"grade": "N/A", "pct_within_5": np.nan,
                "pct_within_10": np.nan, "pct_within_15": np.nan, "n": 0}
    p5 = float(np.mean(abs_err <= 5.0) * 100.0)
    p10 = float(np.mean(abs_err <= 10.0) * 100.0)
    p15 = float(np.mean(abs_err <= 15.0) * 100.0)
    grade = "D"
    for g, r5, r10, r15 in BHS_GRADES:
        if p5 >= r5 and p10 >= r10 and p15 >= r15:
            grade = g
            break
    return {"grade": grade, "pct_within_5": p5, "pct_within_10": p10,
            "pct_within_15": p15, "n": n}


# --------------------------------------------------------------------------
# IEEE 1708 - graded by the mean absolute error (MAE) of the estimate
# --------------------------------------------------------------------------
# MAE = mean(|estimate - reference|), the same quantity the rest of this
# codebase calls MAE_SBP / MAE_DBP.
#     A: MAE <= 5 mmHg     B: <= 6     C: <= 7     D: > 7
def ieee_grade(errors):
    abs_err = np.abs(np.asarray(errors, dtype=np.float64))
    n = abs_err.size
    if n == 0:
        return {"grade": "N/A", "mae": np.nan, "n": 0}
    mae = float(np.mean(abs_err))
    if mae <= 5.0:
        grade = "A"
    elif mae <= 6.0:
        grade = "B"
    elif mae <= 7.0:
        grade = "C"
    else:
        grade = "D"
    return {"grade": grade, "mae": mae, "n": n}


# --------------------------------------------------------------------------
# AAMI / ANSI/AAMI SP10 - mean error and SD of error
# --------------------------------------------------------------------------
#   PASS if |ME| <= 5 mmHg AND SD <= 8 mmHg   (needs >=85 subjects to be valid)
AAMI_ME_LIMIT = 5.0
AAMI_SD_LIMIT = 8.0


def aami_sp10(errors):
    err = np.asarray(errors, dtype=np.float64)
    n = err.size
    if n == 0:
        return {"passed": False, "me": np.nan, "sd": np.nan, "n": 0}
    me = float(np.mean(err))
    sd = float(np.std(err, ddof=1)) if n > 1 else float("nan")
    passed = bool(abs(me) <= AAMI_ME_LIMIT and sd <= AAMI_SD_LIMIT)
    return {"passed": passed, "me": me, "sd": sd, "n": n}


# --------------------------------------------------------------------------
# ISO 81060-2:2018 - two criteria
# --------------------------------------------------------------------------
# Criterion 1: on ALL paired readings, mean error <= 5 mmHg AND SD <= 8 mmHg
#              (identical numbers to AAMI SP10).
# Criterion 2: on the PER-SUBJECT mean errors, the SD of those subject means
#              must be <= a limit that TIGHTENS as |ME| grows. The 2018 standard
#              tabulates that limit (for the reference target of >=85 subjects);
#              we interpolate it. |ME| > 5 => limit 0 => automatic fail.
ISO_ME_LIMIT = 5.0
ISO_SD_LIMIT = 8.0
# (|mean error| mmHg, max allowed SD of per-subject mean errors mmHg)
_ISO_C2_TABLE = (
    (0.0, 6.95), (0.5, 6.87), (1.0, 6.71), (1.5, 6.47), (2.0, 6.15),
    (2.5, 5.76), (3.0, 5.29), (3.5, 4.73), (4.0, 4.08), (4.5, 3.34),
    (5.0, 2.49),
)


def _iso_c2_sd_limit(abs_me):
    if abs_me >= _ISO_C2_TABLE[-1][0]:
        return 0.0 if abs_me > _ISO_C2_TABLE[-1][0] else _ISO_C2_TABLE[-1][1]
    xs = [t[0] for t in _ISO_C2_TABLE]
    ys = [t[1] for t in _ISO_C2_TABLE]
    return float(np.interp(abs_me, xs, ys))


def iso_81060_2(errors, subject_ids):
    err = np.asarray(errors, dtype=np.float64)
    n = err.size
    if n == 0:
        return {"criterion1_pass": False, "criterion2_pass": False,
                "overall_pass": False, "me": np.nan, "sd": np.nan,
                "sd_subject_means": np.nan, "c2_sd_limit": np.nan,
                "n": 0, "n_subjects": 0}

    me = float(np.mean(err))
    sd = float(np.std(err, ddof=1)) if n > 1 else float("nan")
    criterion1 = bool(abs(me) <= ISO_ME_LIMIT and sd <= ISO_SD_LIMIT)

    # Criterion 2: SD across per-subject mean errors.
    subj = np.asarray(subject_ids)
    subj_means = np.array([err[subj == s].mean() for s in pd.unique(subj)],
                          dtype=np.float64)
    n_subjects = subj_means.size
    sd_subject_means = (float(np.std(subj_means, ddof=1))
                        if n_subjects > 1 else float("nan"))
    c2_limit = _iso_c2_sd_limit(abs(me))
    criterion2 = bool(n_subjects > 1 and sd_subject_means <= c2_limit)

    return {"criterion1_pass": criterion1, "criterion2_pass": criterion2,
            "overall_pass": bool(criterion1 and criterion2),
            "me": me, "sd": sd, "sd_subject_means": sd_subject_means,
            "c2_sd_limit": c2_limit, "n": n, "n_subjects": n_subjects}


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def compute_bp_standards(pred, true, subject_ids):
    """All four standards for one BP channel (SBP or DBP).

    pred, true: absolute BP in mmHg (1-D, paired). subject_ids: same length.
    """
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    errors = pred - true
    return {
        "n": int(errors.size),
        "n_subjects": int(pd.unique(np.asarray(subject_ids)).size),
        "me": float(np.mean(errors)) if errors.size else float("nan"),
        "sd": float(np.std(errors, ddof=1)) if errors.size > 1 else float("nan"),
        "mae": float(np.mean(np.abs(errors))) if errors.size else float("nan"),
        "bhs": bhs_grade(errors),
        "ieee": ieee_grade(errors),
        "aami": aami_sp10(errors),
        "iso": iso_81060_2(errors, subject_ids),
    }


def compute_bp_standards_from_df(df, subject_col="Subject_ID"):
    """Convenience wrapper for the LOSO predictions frame with columns
    SBP_true/SBP_pred/DBP_true/DBP_pred (+ Subject_ID). Returns {'SBP':..,'DBP':..}."""
    subj = df[subject_col] if subject_col in df.columns else np.zeros(len(df))
    return {
        "SBP": compute_bp_standards(df["SBP_pred"], df["SBP_true"], subj),
        "DBP": compute_bp_standards(df["DBP_pred"], df["DBP_true"], subj),
    }


def format_standards_report(standards, zero_delta_mae=None):
    """Human-readable report. standards = {'SBP': {...}, 'DBP': {...}}.

    zero_delta_mae (optional): {'SBP': x, 'DBP': y} zero-delta baseline MAE, so
    the report can put the grades next to the bar the model actually has to beat.
    """
    lines = []
    lines.append("=" * 78)
    lines.append("BP DEVICE-VALIDATION STANDARDS (BHS / AAMI SP10 / ISO 81060-2:2018 / IEEE 1708)")
    lines.append("=" * 78)
    any_ch = next(iter(standards.values()))
    lines.append(f"Pooled across all LOSO test readings: n={any_ch['n']} readings, "
                 f"n_subjects={any_ch['n_subjects']}.")
    if any_ch["n_subjects"] < 85:
        lines.append("NOTE: these standards assume >=85 subjects; with fewer, treat pass/fail "
                     "as indicative only.")
    lines.append("Applied to ABSOLUTE BP error (estimate - reference), which for this "
                 "calibrated,")
    lines.append("LOSO-evaluated model still contains the per-subject calibration offset.")
    lines.append("MAE below is the POOLED (reading-weighted) MAE over all readings; it differs")
    lines.append("from the training log's 'Mean MAE', which is the mean of the per-fold MAEs")
    lines.append("(equal weight per fold). SD is the spread of the error - not the MAE.")
    lines.append("")

    for ch in ("SBP", "DBP"):
        s = standards[ch]
        bhs, ieee, aami, iso = s["bhs"], s["ieee"], s["aami"], s["iso"]
        lines.append("-" * 78)
        header = f"{ch}:  MAE(pooled)={s['mae']:.2f}  ME={s['me']:+.2f}  SD={s['sd']:.2f} mmHg"
        if zero_delta_mae is not None and ch in zero_delta_mae:
            header += f"   (zero-delta baseline MAE={zero_delta_mae[ch]:.2f})"
        lines.append(header)
        lines.append("-" * 78)
        lines.append(
            f"  BHS      : Grade {bhs['grade']}   "
            f"(<=5mmHg {bhs['pct_within_5']:.1f}%, <=10mmHg {bhs['pct_within_10']:.1f}%, "
            f"<=15mmHg {bhs['pct_within_15']:.1f}%; A needs 60/85/95)")
        lines.append(
            f"  IEEE 1708: Grade {ieee['grade']}   (MAE "
            f"{ieee['mae']:.2f} mmHg; A<=5, B<=6, C<=7, D>7)")
        lines.append(
            f"  AAMI SP10: {'PASS' if aami['passed'] else 'FAIL'}   "
            f"(|ME|={abs(aami['me']):.2f}<=5? and SD={aami['sd']:.2f}<=8?)")
        lines.append(
            f"  ISO 81060-2:2018:")
        lines.append(
            f"      Criterion 1 (all readings): {'PASS' if iso['criterion1_pass'] else 'FAIL'}   "
            f"(ME={iso['me']:+.2f}<=|5|, SD={iso['sd']:.2f}<=8)")
        lines.append(
            f"      Criterion 2 (subject means): {'PASS' if iso['criterion2_pass'] else 'FAIL'}   "
            f"(SD of {iso['n_subjects']} subject-mean errors={iso['sd_subject_means']:.2f}"
            f" <= limit {iso['c2_sd_limit']:.2f} for |ME|={abs(iso['me']):.2f})")
        lines.append(
            f"      Overall: {'PASS' if iso['overall_pass'] else 'FAIL'}")
    lines.append("=" * 78)
    return "\n".join(lines)


def _pool_predictions(paths):
    """Read one or more prediction CSVs (or a run dir holding fold_*/predictions.csv)
    into a single frame. Returns (df, zero_delta_mae_or_None)."""
    import glob
    import os
    frames, zero_delta = [], None
    for path in paths:
        if os.path.isdir(path):
            csvs = sorted(glob.glob(os.path.join(path, "fold_*", "predictions.csv")))
            if not csvs:
                csvs = sorted(glob.glob(os.path.join(path, "**", "predictions.csv"),
                                        recursive=True))
            frames.extend(pd.read_csv(c) for c in csvs)
            summ = os.path.join(path, "loso_summary.csv")
            if zero_delta is None and os.path.isfile(summ):
                s = pd.read_csv(summ)
                if {"MAE_SBP_baseline_calib", "MAE_DBP_baseline_calib"} <= set(s.columns):
                    zero_delta = {"SBP": float(s["MAE_SBP_baseline_calib"].mean()),
                                  "DBP": float(s["MAE_DBP_baseline_calib"].mean())}
        else:
            frames.append(pd.read_csv(path))
    if not frames:
        raise SystemExit("No predictions.csv found in: " + ", ".join(paths))
    return pd.concat(frames, axis=0, ignore_index=True), zero_delta


def standards_summary_rows(standards):
    """Flat rows for CSV export: one row per channel."""
    rows = []
    for ch in ("SBP", "DBP"):
        s = standards[ch]
        rows.append({
            "channel": ch, "n": s["n"], "n_subjects": s["n_subjects"],
            "ME": s["me"], "SD": s["sd"], "MAE": s["mae"],
            "BHS_grade": s["bhs"]["grade"],
            "BHS_pct_within_5": s["bhs"]["pct_within_5"],
            "BHS_pct_within_10": s["bhs"]["pct_within_10"],
            "BHS_pct_within_15": s["bhs"]["pct_within_15"],
            "IEEE_grade": s["ieee"]["grade"], "IEEE_MAE": s["ieee"]["mae"],
            "AAMI_SP10_pass": s["aami"]["passed"],
            "ISO_criterion1_pass": s["iso"]["criterion1_pass"],
            "ISO_criterion2_pass": s["iso"]["criterion2_pass"],
            "ISO_overall_pass": s["iso"]["overall_pass"],
            "ISO_sd_subject_means": s["iso"]["sd_subject_means"],
            "ISO_c2_sd_limit": s["iso"]["c2_sd_limit"],
        })
    return rows


def main():
    """CLI: grade existing LOSO predictions without retraining.

        python3 bp_standards.py <run_dir | predictions.csv> [more ...] [--save]

    Pass a run directory (pools its fold_*/predictions.csv and reads
    loso_summary.csv for the zero-delta baseline) or explicit CSV files.
    """
    import argparse
    import os

    ap = argparse.ArgumentParser(description=main.__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+",
                    help="Run directory (with fold_*/predictions.csv) or CSV file(s).")
    ap.add_argument("--save", action="store_true",
                    help="Also write bp_standards.txt/.csv next to the first path.")
    args = ap.parse_args()

    df, zero_delta = _pool_predictions(args.paths)
    standards = compute_bp_standards_from_df(df)
    report = format_standards_report(standards, zero_delta)
    print(report)

    if args.save:
        out_base = args.paths[0] if os.path.isdir(args.paths[0]) else os.path.dirname(args.paths[0]) or "."
        with open(os.path.join(out_base, "bp_standards.txt"), "w") as f:
            f.write(report + "\n")
        pd.DataFrame(standards_summary_rows(standards)).to_csv(
            os.path.join(out_base, "bp_standards.csv"), index=False)
        print(f"\nWrote bp_standards.txt and bp_standards.csv to {out_base}")


if __name__ == "__main__":
    main()
