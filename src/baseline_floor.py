"""
No-model diagnostic: compute the LOSO "floor" MAE that any model has to beat,
using the exact same fold definitions as train_worker.py's LOSOTrainingWorker,
but without instantiating/training a network at all. Two baselines, per fold:

  pop-mean     - predict the TRAIN pool's mean SBP/DBP for every window of the
                 held-out subject. Uses no per-subject info at all.
  zero-delta   - predict the held-out subject's OWN calibration reading (mean
                 of their recording's first `calib_windows` windows) for every
                 later window, i.e. assume BP never changes from calibration.
                 This is the real bar a calibration-relative model must clear.

Run this before spending GPU/CPU time training the real CNN-BiLSTM, so you
know upfront how good "doing nothing clever" already looks.

Usage:
    python baseline_floor.py --combined-dir feature_result/combined
"""
import argparse

import numpy as np
import pandas as pd

from skna_dataset import SKNASequenceDataset


def compute_floor(combined_dir, seq_len=10, stride=1, calib_windows=10, val_frac=0.15):
    ds = SKNASequenceDataset(
        raw_signal_npz=f"{combined_dir}/all_raw_signal.npz",
        raw_signal_labels_csv=f"{combined_dir}/all_raw_signal_labels.csv",
        skna_signal_npz=f"{combined_dir}/all_skna_signal.npz",
        iskna_rskna_npz=f"{combined_dir}/all_iskna_rskna.npz",
        askna_npz=f"{combined_dir}/all_askna.npz",
        seq_len=seq_len, stride=stride, calib_windows=calib_windows,
    )
    groups = ds.sequence_groups()
    subjects = sorted(set(groups))
    print(f"Loaded {len(ds)} sequences across {len(subjects)} subjects: {subjects}")

    rows = []
    for test_subject in subjects:
        remaining = [s for s in subjects if s != test_subject]
        train_seq_idx, _val_seq_idx = ds.split_within_subjects(remaining, val_frac=val_frac)
        test_seq_idx = np.where(groups == test_subject)[0]

        train_pos = ds.seq_index[train_seq_idx][:, -1]
        test_pos = ds.seq_index[test_seq_idx][:, -1]
        labels = ds._arrays["labels"]

        train_abs = labels[["SBP", "DBP"]].iloc[train_pos].to_numpy()
        test_abs = labels[["SBP", "DBP"]].iloc[test_pos].to_numpy()
        test_baseline = np.stack([
            ds._arrays["baseline_sbp"][test_pos],
            ds._arrays["baseline_dbp"][test_pos],
        ], axis=1)
        test_delta = test_abs - test_baseline

        train_mean = train_abs.mean(axis=0)
        popmean_mae = np.abs(test_abs - train_mean).mean(axis=0)
        zerodelta_mae = np.abs(test_delta).mean(axis=0)

        rows.append({
            "Subject_ID": test_subject, "n_test_seq": len(test_seq_idx),
            "MAE_SBP_popmean": popmean_mae[0], "MAE_DBP_popmean": popmean_mae[1],
            "MAE_SBP_zerodelta": zerodelta_mae[0], "MAE_DBP_zerodelta": zerodelta_mae[1],
        })

    summary = pd.DataFrame(rows)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--combined-dir", default="feature_result/combined")
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--calib-windows", type=int, default=10)
    parser.add_argument("--val-frac", type=float, default=0.15)
    args = parser.parse_args()

    summary = compute_floor(args.combined_dir, args.seq_len, args.stride,
                             args.calib_windows, args.val_frac)
    print()
    print(summary.to_string(index=False))
    print()
    print(f"Mean MAE_SBP: pop-mean={summary['MAE_SBP_popmean'].mean():.2f}  "
          f"zero-delta={summary['MAE_SBP_zerodelta'].mean():.2f}")
    print(f"Mean MAE_DBP: pop-mean={summary['MAE_DBP_popmean'].mean():.2f}  "
          f"zero-delta={summary['MAE_DBP_zerodelta'].mean():.2f}")
