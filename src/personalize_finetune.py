"""
Per-subject fine-tuning (personalization) experiment for SKNA -> BP.

For each held-out subject that has a LOSO base checkpoint (fold_<subj>/checkpoint.pt,
trained WITHOUT that subject), we:

  1. Split THAT subject's own recording by TIME into three temporally-disjoint
     buckets (windows straddling a boundary are dropped -> zero window overlap):
        adapt-train : [0.0, 0.5)   -- the "calibration" slice the device would see
        adapt-val   : [0.5, 0.6)   -- early-stopping signal for the fine-tune
        test        : [0.6, 1.0]   -- held-out evaluation, never seen while adapting
  2. Evaluate the base (population) model on the test slice  -> PRE-FT.
  3. Freeze everything except the last LSTM layer + attention pool + head
     (freeze_for_hybrid_calibration, McBP-Net style) and fine-tune on adapt-train,
     early-stopping on adapt-val.
  4. Evaluate the fine-tuned model on the SAME test slice     -> POST-FT.
  5. Compare both against the ZERO-DELTA baseline on that slice (predict the
     subject's own calibration reading, i.e. no BP change) -- the real bar.

Normalization (norm_mode='calib') and the BP baseline are per-recording, computed
only from each recording's own first `calib_windows` windows, so nothing about the
test slice leaks into adaptation beyond the one-time calibration a real cuffless
device would also have.

Reuses the exact architecture + target scaler stored in each checkpoint, so PRE-FT
reproduces that fold's LOSO behaviour (restricted to the test slice).

Run (from repo root):
    /home/rrpzy/venv/bin/python src/personalize_finetune.py
"""
import os
import sys
import json
import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from skna_dataset import SKNASequenceDataset, three_way_time_split
from model import (SKNABPModel, TargetScaler, EarlyStopper, train_one_epoch,
                   evaluate, freeze_for_hybrid_calibration, ANN_POOL_LEN)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The base LOSO checkpoints were trained on the full 14-subject aggregate; the
# old feature_result/combined was only a 6-subject subset (now deleted), so the
# 5sWindow aggregate is the correct source. Override with --data-dir.
DEFAULT_DATA = os.path.join(REPO, "feature_result", "5sWindow", "5sWindow Dataset")
CKPT_DIR = os.path.join(REPO, "training_results")
GROUP_COLS = ("Subject_ID", "Session_ID")


def zero_delta_baseline(test_metrics):
    """MAE of always predicting 'no change from the subject's own calibration'
    (pred_delta == 0) on the test slice -- the real bar the model must beat."""
    td = test_metrics["true_delta"]
    return float(np.abs(td[:, 0]).mean()), float(np.abs(td[:, 1]).mean())


def build_dataset(params, data_dir):
    use_fd = params.get("use_fd", False) or params.get("use_fft", False) or params.get("use_psd", False)
    use_cwt = params.get("use_cwt", False)
    fd_npz = os.path.join(data_dir, "all_fd_signal.npz") if use_fd else None
    cwt_npz = os.path.join(data_dir, "all_cwt_signal.npz") if use_cwt else None
    return SKNASequenceDataset(
        raw_signal_npz=os.path.join(data_dir, "all_raw_signal.npz"),
        raw_signal_labels_csv=os.path.join(data_dir, "all_raw_signal_labels.csv"),
        skna_signal_npz=os.path.join(data_dir, "all_skna_signal.npz"),
        iskna_rskna_npz=os.path.join(data_dir, "all_iskna_rskna.npz"),
        askna_npz=os.path.join(data_dir, "all_askna.npz"),
        fd_signal_npz=fd_npz, cwt_signal_npz=cwt_npz,
        seq_len=params.get("seq_len", 12), stride=params.get("stride", 5),
        calib_windows=params.get("calib_windows", 10),
        norm_mode=params.get("norm_mode", "calib"),
        downsample=int(params.get("downsample", 1) or 1),
    )


def build_model(params, device):
    return SKNABPModel(
        ecg_channels=tuple(params["ecg_channels"]),
        skna_channels=tuple(params["skna_channels"]),
        iskna_rskna_channels=tuple(params["iskna_rskna_channels"]),
        lstm_hidden=params.get("lstm_hidden", 64),
        lstm_layers=params.get("lstm_layers", 2),
        dropout=params.get("dropout", 0.3),
        use_ecg=params.get("use_ecg", True), use_skna=params.get("use_skna", True),
        use_iskna_rskna=params.get("use_iskna_rskna", True),
        use_askna=params.get("use_askna", True),
        use_fd=params.get("use_fd", False), fd_channels=tuple(params.get("fd_channels", (16, 32, 64))),
        use_fft=params.get("use_fft", False), fft_channels=tuple(params.get("fft_channels", (16, 32, 64))),
        use_psd=params.get("use_psd", False), psd_channels=tuple(params.get("psd_channels", (16, 32, 64))),
        use_cwt=params.get("use_cwt", False), cwt_channels=tuple(params.get("cwt_channels", (8, 16, 32))),
        cnn_checkpoint_above=params.get("cnn_checkpoint_above", 256),
        cnn_arch=params.get("cnn_arch", "plain"),
        # Absent (pre-cwt_arch checkpoints) => follow cnn_arch, as they were built.
        cwt_arch=params.get("cwt_arch"),
        resnet_blocks_per_stage=params.get("resnet_blocks_per_stage", 1),
        ann_pool_len=params.get("ann_pool_len", ANN_POOL_LEN),
    ).to(device)


def run(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_dir = args.ckpt_dir

    # Discover subjects that have a base checkpoint.
    subjects = []
    for name in sorted(os.listdir(ckpt_dir)):
        d = os.path.join(ckpt_dir, name)
        if name.startswith("fold_") and os.path.isfile(os.path.join(d, "checkpoint.pt")):
            subjects.append(name[len("fold_"):])
    if args.subjects:
        wanted = set(args.subjects.split(","))
        subjects = [s for s in subjects if s in wanted]
    print(f"[personalize] device={device}  ckpt_dir: {ckpt_dir}")
    print(f"[personalize] subjects with checkpoints: {subjects}", flush=True)

    # All checkpoints share the same preprocessing params, so build the dataset once
    # from the first checkpoint's params and reuse it for every subject.
    ref = torch.load(os.path.join(ckpt_dir, f"fold_{subjects[0]}", "checkpoint.pt"),
                     map_location="cpu", weights_only=False)
    ds = build_dataset(ref["params"], args.data_dir)
    print(f"[personalize] data_dir: {args.data_dir}")
    print(f"[personalize] dataset: {len(ds)} sequences, "
          f"{len(set(ds.sequence_groups()))} subjects", flush=True)

    rows = []
    os.makedirs(args.out_dir, exist_ok=True)
    for subj in subjects:
        ckpt_path = os.path.join(ckpt_dir, f"fold_{subj}", "checkpoint.pt")
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        p = ck["params"]
        if subj in (ck.get("train_subjects") or []):
            print(f"[{subj}] SKIP: checkpoint was trained ON this subject (leakage).", flush=True)
            continue

        adapt_idx, val_idx, test_idx = three_way_time_split(
            ds, subj, adapt_end=args.adapt_end, val_end=args.val_end)
        if len(adapt_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
            print(f"[{subj}] SKIP: split too small "
                  f"(adapt={len(adapt_idx)}, val={len(val_idx)}, test={len(test_idx)}).",
                  flush=True)
            continue

        adapt_ds = ds.subset(adapt_idx)
        val_ds = ds.subset(val_idx)
        test_ds = ds.subset(test_idx)
        bs = p.get("batch_size", 32)
        adapt_loader = DataLoader(adapt_ds, batch_size=bs, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False)

        target_scaler = TargetScaler.from_state_dict(ck["target_scaler"])

        # ---- PRE-FT: base (population) model on the test slice ----
        model = build_model(p, device)
        model.load_state_dict(ck["model_state"])
        pre = evaluate(model, test_loader, device, target_scaler, return_predictions=True)
        bl_sbp, bl_dbp = zero_delta_baseline(pre)

        # ---- Fine-tune (hybrid calibration): last LSTM layer + attn + head ----
        freeze_for_hybrid_calibration(model)
        trainable = [q for q in model.parameters() if q.requires_grad]
        n_train = sum(q.numel() for q in trainable)
        opt = torch.optim.Adam(trainable, lr=args.lr, weight_decay=p.get("weight_decay", 1e-4))
        stopper = EarlyStopper(patience=args.patience, min_delta=0.01)
        best_state, best_epoch, best_sel = None, -1, float("inf")
        for epoch in range(args.epochs):
            tr = train_one_epoch(model, adapt_loader, opt, device, target_scaler)
            vm = evaluate(model, val_loader, device, target_scaler)
            sel = vm["MAE_SBP"] + vm["MAE_DBP"]
            improved = stopper.step(sel)
            if improved:
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_epoch, best_sel = epoch, sel
            if stopper.should_stop:
                break
        if best_state is not None:
            model.load_state_dict(best_state)

        # ---- POST-FT: fine-tuned model on the SAME test slice ----
        post = evaluate(model, test_loader, device, target_scaler, return_predictions=True)

        row = {
            "subject": subj,
            "n_adapt": len(adapt_idx), "n_val": len(val_idx), "n_test": len(test_idx),
            "trainable_params": n_train, "best_epoch": best_epoch,
            "pre_MAE_SBP": pre["MAE_SBP"], "pre_MAE_DBP": pre["MAE_DBP"],
            "post_MAE_SBP": post["MAE_SBP"], "post_MAE_DBP": post["MAE_DBP"],
            "baseline_MAE_SBP": bl_sbp, "baseline_MAE_DBP": bl_dbp,
            "pre_corr_SBP_delta": pre["corr_SBP_delta"], "post_corr_SBP_delta": post["corr_SBP_delta"],
            "pre_corr_DBP_delta": pre["corr_DBP_delta"], "post_corr_DBP_delta": post["corr_DBP_delta"],
            "post_beats_baseline": bool(post["MAE_SBP"] < bl_sbp and post["MAE_DBP"] < bl_dbp),
            "post_beats_pre": bool(post["MAE_SBP"] < pre["MAE_SBP"] and post["MAE_DBP"] < pre["MAE_DBP"]),
        }
        rows.append(row)
        print(f"[{subj}] test={len(test_idx)}  "
              f"PRE  MAE {pre['MAE_SBP']:.2f}/{pre['MAE_DBP']:.2f}  "
              f"POST MAE {post['MAE_SBP']:.2f}/{post['MAE_DBP']:.2f}  "
              f"BASELINE {bl_sbp:.2f}/{bl_dbp:.2f}  "
              f"(post beats baseline: {row['post_beats_baseline']}, "
              f"post beats pre: {row['post_beats_pre']}, best_epoch={best_epoch})",
              flush=True)

    if not rows:
        print("[personalize] No subjects produced results.", flush=True)
        return

    df = pd.DataFrame(rows)
    out_csv = os.path.join(args.out_dir, "personalization_summary.csv")
    df.to_csv(out_csv, index=False)

    n = len(df)
    print("\n" + "=" * 78)
    print(f"PERSONALIZATION SUMMARY  ({n} subjects)")
    print("=" * 78)
    print(df.to_string(index=False))
    print("-" * 78)
    print(f"Mean PRE  MAE : SBP {df['pre_MAE_SBP'].mean():.2f}  DBP {df['pre_MAE_DBP'].mean():.2f}")
    print(f"Mean POST MAE : SBP {df['post_MAE_SBP'].mean():.2f}  DBP {df['post_MAE_DBP'].mean():.2f}")
    print(f"Mean BASELINE : SBP {df['baseline_MAE_SBP'].mean():.2f}  DBP {df['baseline_MAE_DBP'].mean():.2f}")
    print(f"Post-FT beats zero-delta baseline (SBP & DBP): {int(df['post_beats_baseline'].sum())}/{n}")
    print(f"Post-FT beats pre-FT (SBP & DBP)             : {int(df['post_beats_pre'].sum())}/{n}")
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DEFAULT_DATA, dest="data_dir",
                    help="aggregate dataset dir (default: 5sWindow Dataset)")
    ap.add_argument("--ckpt-dir", default=CKPT_DIR, dest="ckpt_dir",
                    help="dir containing fold_<subject>/checkpoint.pt (default: training_results)")
    ap.add_argument("--subjects", default="", help="comma-separated subset, e.g. s1,s2 (default: all with checkpoints)")
    ap.add_argument("--adapt-end", type=float, default=0.5, dest="adapt_end")
    ap.add_argument("--val-end", type=float, default=0.6, dest="val_end")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=os.path.join(CKPT_DIR, "personalization"), dest="out_dir")
    run(ap.parse_args())
