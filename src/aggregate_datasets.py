"""
Aggregate per-recording labeled CSVs (produced by the GUI pipeline, one run
per subject/session) into single master CSVs ready for model training.

Expected layout (this is what worker.py now produces automatically):

    <root>/
      S01/windowing_labeling/raw_signal_with_labels.csv
      S01/windowing_labeling/skna_signal_with_labels.csv
      S01/windowing_labeling/iskna_rskna_with_labels.csv
      S01/windowing_labeling/askna_with_labels.csv
      S02_session1/windowing_labeling/raw_signal_with_labels.csv
      S02_session2/windowing_labeling/raw_signal_with_labels.csv
      ...

Each of those CSVs already carries Subject_ID / Session_ID columns (stamped
by save_labeled_datasets), so concatenating them is safe and traceable.

Usage (CLI):
    python aggregate_datasets.py --root feature_result --out feature_result/combined

Usage (from a training notebook/script):
    from aggregate_datasets import aggregate_all, get_loso_groups

    combined = aggregate_all("feature_result", out_dir="feature_result/combined")
    ecg_df = combined["raw_signal"]
    groups = get_loso_groups(ecg_df)   # -> array of Subject_ID per row, for
                                        #    sklearn's GroupKFold / LeaveOneGroupOut
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

# The 4 dataset "kinds" produced per recording, and the filename each lives in.
DATASET_FILES = {
    "raw_signal": "raw_signal_with_labels.csv",
    "skna_signal": "skna_signal_with_labels.csv",
    "iskna_rskna": "iskna_rskna_with_labels.csv",
    "askna": "askna_with_labels.csv",
}

# Same 4 kinds, but for the compact .npz layout (labels.csv shared, one .npz
# per kind holding the raw sample arrays). The last two (fd_signal, cwt_signal)
# are OPTIONAL extra branches - only produced when their pipeline checkboxes
# were on - so aggregation of those two is best-effort (a recording missing
# them is skipped with a warning rather than failing the whole run).
NPZ_DATASET_FILES = {
    "raw_signal": "raw_signal.npz",
    "skna_signal": "skna_signal.npz",
    "iskna_rskna": "iskna_rskna.npz",
    "askna": "askna.npz",
    "fd_signal": "fd_signal.npz",
    "cwt_signal": "cwt_signal.npz",
}


def _find_recording_dirs(root_or_roots, labels_subdir="windowing_labeling"):
    """
    Return every '<root>/<recording>/<labels_subdir>' folder that exists.
    root_or_roots can be a single path or a list of paths, in case each
    subject's output landed under a different top-level folder rather than
    one shared parent.
    """
    roots = [root_or_roots] if isinstance(root_or_roots, str) else list(root_or_roots)
    found = []
    for root in roots:
        pattern = os.path.join(root, "*", labels_subdir)
        found.extend(d for d in glob.glob(pattern) if os.path.isdir(d))
    return sorted(set(found))


def aggregate_one_kind(root, kind, labels_subdir="windowing_labeling", out_path=None):
    """
    Concatenate one dataset kind (e.g. 'raw_signal') across every recording
    folder under root (root can be a single path or a list of paths, if your
    subjects' outputs live under different top-level folders). Rows are tagged
    with Subject_ID/Session_ID already (from save_labeled_datasets), and a
    Recording_Dir column is added too so you can always trace a row back to
    its source folder if something looks off.
    """
    filename = DATASET_FILES[kind]
    frames = []
    for rec_dir in _find_recording_dirs(root, labels_subdir):
        csv_path = os.path.join(rec_dir, filename)
        if not os.path.isfile(csv_path):
            print(f"[skip] {csv_path} not found.")
            continue
        df = pd.read_csv(csv_path)
        if "Subject_ID" not in df.columns:
            print(f"[warning] {csv_path} has no Subject_ID column — it was "
                  f"probably produced before subject tagging was added, or "
                  f"the Subject ID field was left blank when running the GUI. "
                  f"Re-run that recording through the GUI with a Subject ID set.")
        df["Recording_Dir"] = os.path.basename(os.path.dirname(rec_dir))
        df = df.copy()  # de-fragment after column assignment on a wide frame
        frames.append(df)

    if not frames:
        roots_str = root if isinstance(root, str) else ", ".join(root)
        raise FileNotFoundError(
            f"No '{filename}' files found under any '{roots_str}'/*/{labels_subdir}/ folder.")

    combined = pd.concat(frames, axis=0, ignore_index=True)

    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        combined.to_csv(out_path, index=False)
        n_subj = combined["Subject_ID"].nunique() if "Subject_ID" in combined.columns else "?"
        print(f"[saved] {out_path}: {combined.shape[0]} rows x {combined.shape[1]} cols "
              f"from {len(frames)} recording(s), {n_subj} subject(s).")

    return combined


def aggregate_all(root, labels_subdir="windowing_labeling", out_dir=None):
    """
    Aggregate all 4 dataset kinds (wide-CSV format). Returns a dict
    {kind: combined_DataFrame}. If out_dir is given, also writes
    'all_<kind>_with_labels.csv' there.
    """
    results = {}
    for kind in DATASET_FILES:
        out_path = os.path.join(out_dir, f"all_{kind}_with_labels.csv") if out_dir else None
        try:
            results[kind] = aggregate_one_kind(root, kind, labels_subdir, out_path)
        except FileNotFoundError as e:
            print(f"[warning] {e}")
    return results


def aggregate_one_kind_npz(root, kind, labels_subdir="windowing_labeling", out_dir=None):
    """
    npz equivalent of aggregate_one_kind(): concatenates each recording's
    labels.csv (metadata) and '<kind>.npz' (sample arrays) across every
    recording folder, in matching row order. Returns (combined_meta_df,
    dict of combined arrays). If out_dir is given, writes
    'all_<kind>_labels.csv' and 'all_<kind>.npz'.
    """
    filename = NPZ_DATASET_FILES[kind]
    meta_frames = []
    array_chunks = {}
    skipped_misaligned = []

    for rec_dir in _find_recording_dirs(root, labels_subdir):
        rec_name = os.path.basename(os.path.dirname(rec_dir))
        npz_path = os.path.join(rec_dir, filename)
        meta_path = os.path.join(rec_dir, "labels.csv")
        if not (os.path.isfile(npz_path) and os.path.isfile(meta_path)):
            print(f"[skip] {rec_dir}: missing '{filename}' or 'labels.csv'.")
            continue
        meta = pd.read_csv(meta_path)
        npz = np.load(npz_path)
        # Per-recording alignment guard. Every array in the .npz must have exactly
        # one row per labels.csv row; otherwise concatenating it would shift every
        # LATER recording's rows out of sync with its labels (a partial file from a
        # cancelled run — e.g. a truncated cwt_signal.npz — corrupts the whole
        # combined array while only tripping a single row-count warning at the end).
        # Exclude such a recording entirely for this kind so the rest stays aligned.
        misaligned = {k: int(npz[k].shape[0]) for k in npz.files if npz[k].shape[0] != len(meta)}
        if misaligned:
            print(f"[skip] {rec_name}: '{filename}' array rows {misaligned} != labels.csv "
                  f"rows ({len(meta)}) — this recording's '{kind}' arrays are partial/misaligned "
                  f"(e.g. a cancelled run) and are being EXCLUDED so the combined '{kind}' "
                  f"dataset stays row-aligned. Re-process this recording to include it.")
            skipped_misaligned.append(rec_name)
            continue
        meta["Recording_Dir"] = rec_name
        meta_frames.append(meta)
        for key in npz.files:
            array_chunks.setdefault(key, []).append(npz[key])

    if not meta_frames:
        roots_str = root if isinstance(root, str) else ", ".join(root)
        raise FileNotFoundError(
            f"No '{filename}' + labels.csv pairs found under '{roots_str}'/*/{labels_subdir}/.")

    combined_meta = pd.concat(meta_frames, axis=0, ignore_index=True)
    combined_arrays = {k: np.concatenate(v, axis=0) for k, v in array_chunks.items()}

    for key, arr in combined_arrays.items():
        if len(arr) != len(combined_meta):
            print(f"[warning] '{kind}' array '{key}' has {len(arr)} rows but the combined "
                  f"metadata has {len(combined_meta)} rows — something is misaligned.")

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        combined_meta.to_csv(os.path.join(out_dir, f"all_{kind}_labels.csv"), index=False)
        npz_out = os.path.join(out_dir, f"all_{kind}.npz")
        np.savez_compressed(npz_out, **combined_arrays)
        size_mb = os.path.getsize(npz_out) / 1e6
        n_subj = combined_meta["Subject_ID"].nunique() if "Subject_ID" in combined_meta.columns else "?"
        excluded = (f" ({len(skipped_misaligned)} recording(s) EXCLUDED for misalignment: "
                    f"{', '.join(skipped_misaligned)})") if skipped_misaligned else ""
        print(f"[saved] all_{kind}.npz ({size_mb:.1f} MB) + all_{kind}_labels.csv: "
              f"{len(combined_meta)} rows from {len(meta_frames)} recording(s), {n_subj} subject(s).{excluded}")

    return combined_meta, combined_arrays


def aggregate_all_npz(root, labels_subdir="windowing_labeling", out_dir=None):
    """npz equivalent of aggregate_all(). Returns {kind: (meta_df, arrays_dict)}."""
    results = {}
    for kind in NPZ_DATASET_FILES:
        try:
            results[kind] = aggregate_one_kind_npz(root, kind, labels_subdir, out_dir)
        except FileNotFoundError as e:
            print(f"[warning] {e}")
    return results


def load_npz_dataset(labels_csv, npz_path, array_key="X"):
    """
    Convenience loader for a training script: reads one aggregated npz dataset
    and returns (X, sbp, dbp, groups) ready for model.fit()/GroupKFold.
    array_key is the array name inside the npz: 'X' for raw_signal/
    skna_signal/askna, or 'rSKNA'/'iSKNA' for the combined iskna_rskna dataset.
    """
    meta = pd.read_csv(labels_csv)
    X = np.load(npz_path)[array_key]
    return X, meta["SBP"].to_numpy(), meta["DBP"].to_numpy(), get_loso_groups(meta)


def get_loso_groups(df, group_col="Subject_ID"):
    """
    Return the per-row group array needed for subject-wise / leave-one-subject-out
    cross-validation (e.g. sklearn.model_selection.GroupKFold or
    LeaveOneGroupOut). Raises if the column is missing or contains any blank/NaN
    values, since silently falling back to something else — or letting NaN
    slip through as a "group" — would risk leaking subjects across folds.
    """
    if group_col not in df.columns:
        raise ValueError(
            f"'{group_col}' column not found — make sure every recording was "
            f"processed through the GUI with a Subject ID set before aggregating."
        )
    groups = df[group_col]
    if groups.isna().any() or (groups.astype(str).str.strip() == "").any():
        raise ValueError(
            f"'{group_col}' has missing/blank values for some rows — check which "
            f"recording(s) were run without a Subject ID and re-process them."
        )
    return groups.to_numpy()


def summarize_subjects(df, group_col="Subject_ID", session_col="Session_ID"):
    """Quick sanity-check table: window count and duplicate-recording check per subject."""
    cols = [c for c in (group_col, session_col) if c in df.columns]
    if not cols:
        raise ValueError(f"Neither '{group_col}' nor '{session_col}' found in the dataframe.")
    # Blank Session_ID gets written/read back as NaN, and groupby drops NaN
    # groups by default — fill it so single-session subjects aren't silently
    # dropped from the summary.
    df = df.copy()
    for c in cols:
        df[c] = df[c].fillna("")
    return df.groupby(cols, dropna=False).size().rename("num_windows").reset_index()


def flag_short_recordings(df, group_col="Subject_ID", session_col="Session_ID",
                           relative_threshold=0.5):
    """
    Flag recordings whose window count is well below the rest (e.g. your 10-min
    session vs the 30-min ones). Doesn't fix anything automatically — upsampling
    a short recording to match the others would just duplicate correlated
    windows and inflate confidence in that subject's fold. This is purely so the
    imbalance is visible before you run LOSO CV and interpret per-fold metrics.

    relative_threshold: a recording is flagged if its window count is below
    relative_threshold * median(window count across all recordings).
    """
    summary = summarize_subjects(df, group_col, session_col)
    median_windows = summary["num_windows"].median()
    summary["is_short_recording"] = summary["num_windows"] < (relative_threshold * median_windows)
    if summary["is_short_recording"].any():
        short = summary[summary["is_short_recording"]]
        print(f"[note] {len(short)} recording(s) have < {relative_threshold:.0%} of the "
              f"median window count ({median_windows:.0f}). When these are the held-out "
              f"subject in LOSO CV, expect a noisier (higher-variance) fold metric — "
              f"report per-subject scores individually rather than only the average:")
        print(short.to_string(index=False))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, nargs="+",
                         help="Root output dir(s) containing one subfolder per recording "
                              "(what you set as 'Output dir' in the GUI). Pass multiple "
                              "paths if different subjects were saved under different "
                              "top-level folders.")
    parser.add_argument("--labels-subdir", default="windowing_labeling",
                         help="Name of the labeled-dataset subfolder inside each recording dir.")
    parser.add_argument("--out", default=None,
                         help="Directory to write the combined output into "
                              "(default: '<first root>/combined').")
    parser.add_argument("--format", choices=["npz", "csv"], default="npz",
                         help="Which per-recording output format to aggregate: 'npz' "
                              "(labels.csv + compact .npz arrays, current GUI default) "
                              "or 'csv' (legacy wide *_with_labels.csv files). Default: npz.")
    args = parser.parse_args()

    out_dir = args.out or os.path.join(args.root[0], "combined")
    roots = args.root if len(args.root) > 1 else args.root[0]

    if args.format == "npz":
        combined = aggregate_all_npz(roots, labels_subdir=args.labels_subdir, out_dir=out_dir)
        if "raw_signal" in combined:
            print("\nWindows per subject/session (from raw_signal):")
            flag_short_recordings(combined["raw_signal"][0])
    else:
        combined = aggregate_all(roots, labels_subdir=args.labels_subdir, out_dir=out_dir)
        if "raw_signal" in combined:
            print("\nWindows per subject/session (from raw_signal):")
            flag_short_recordings(combined["raw_signal"])