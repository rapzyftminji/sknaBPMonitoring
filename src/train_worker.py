import os
import traceback

import numpy as np
import pandas as pd
from PyQt5.QtCore import QThread, pyqtSignal

class LOSOTrainingWorker(QThread):
    log = pyqtSignal(str)
    fold_started = pyqtSignal(str)
    epoch_progress = pyqtSignal(str, int, float, float, float)  # subject, epoch, train_loss, val_loss, val_select_mmhg
    fold_finished = pyqtSignal(str, object)                # subject, fold_result dict
    all_finished = pyqtSignal(object)                       # {"summary": df, "folds": {...}}
    error = pyqtSignal(str)

    def __init__(self, params, parent=None):
        super().__init__(parent)
        self.params = params
        self._stop_requested = False

    def request_stop(self):
        self._stop_requested = True

    def _should_stop(self):
        return self._stop_requested

    def run(self):
        try:
            try:
                import torch
                from torch.utils.data import DataLoader
                from skna_dataset import SKNASequenceDataset, three_way_time_split
                from model import (SKNABPModel, TargetScaler, EarlyStopper, train_one_epoch,
                                   evaluate, CNN_CHECKPOINT_ABOVE, ANN_POOL_LEN,
                                   freeze_for_hybrid_calibration)
            except (OSError, ImportError) as e:
                self.error.emit(
                    "Failed to load PyTorch (this is an environment/install issue, "
                    "not a bug in this pipeline's code):\n\n" + str(e) +
                    "\n\nThis is almost always one of:\n"
                    "  1. PyTorch isn't installed in this environment - run: "
                    "pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
                    "  2. Missing Microsoft Visual C++ Redistributable (x64) on Windows - "
                    "install the latest from https://aka.ms/vs/17/release/vc_redist.x64.exe "
                    "and reboot.\n"
                    "  3. A CUDA build of torch installed on a machine without a matching "
                    "NVIDIA driver/CUDA version - try: pip uninstall torch, then "
                    "pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
                    "  4. A corrupted install - try: pip uninstall torch, then reinstall.\n"
                    "The rest of the app (preprocessing tabs) is unaffected by this."
                )
                return

            p = self.params
            combined_dir = p["combined_dir"]
            out_dir = p.get("output_dir") or os.path.join(combined_dir, "training_results")
            os.makedirs(out_dir, exist_ok=True)

            seq_len = p.get("seq_len", 10)
            stride = p.get("stride", 1)
            downsample = int(p.get("downsample", 1) or 1)
            p["downsample"] = downsample
            batch_size = p.get("batch_size", 16)
            epochs = p.get("epochs", 50)
            lr = p.get("lr", 1e-3)
            patience = p.get("patience", 10)
            lstm_hidden = p.get("lstm_hidden", 64)
            lstm_layers = p.get("lstm_layers", 2)
            cnn_channels = tuple(p.get("cnn_channels", (16, 32, 64)))
            ecg_channels = tuple(p.get("ecg_channels", cnn_channels))
            skna_channels = tuple(p.get("skna_channels", cnn_channels))
            iskna_rskna_channels = tuple(p.get("iskna_rskna_channels", cnn_channels))
            cnn_arch = p.get("cnn_arch", "plain")
            if cnn_arch not in ("plain", "resnet", "resnet18", "ann"):
                raise ValueError("cnn_arch must be 'plain', 'resnet', 'resnet18', or 'ann', "
                                 f"got {cnn_arch!r}.")
            p["cnn_arch"] = cnn_arch
            # Architecture of the 2D (CWT scalogram) branch, chosen independently of
            # the 1D branches so e.g. ANN-on-waveforms + 2D ResNet-18-on-scalogram is
            # a valid combination. None/absent = follow cnn_arch (what every older
            # param dict meant), except under "ann" (1D only) where it means plain 2D.
            cwt_arch = p.get("cwt_arch") or ("plain" if cnn_arch == "ann" else cnn_arch)
            if cwt_arch not in ("plain", "resnet", "resnet18"):
                raise ValueError(f"cwt_arch must be 'plain', 'resnet', or 'resnet18', got {cwt_arch!r}.")
            p["cwt_arch"] = cwt_arch
            # Depth of the "resnet" branches (ignored by "plain"/"ann"; "resnet18"
            # fixes its own 2-per-stage depth internally). ResNet-18 = 2 blocks/stage.
            resnet_blocks_per_stage = p.get("resnet_blocks_per_stage", 1)
            p["resnet_blocks_per_stage"] = resnet_blocks_per_stage
            # Length the ANN branches adaptive-avg-pool each window down to before
            # flattening into the MLP (ignored by the conv archs).
            ann_pool_len = int(p.get("ann_pool_len", ANN_POOL_LEN))
            p["ann_pool_len"] = ann_pool_len

            p["lstm_layers"] = lstm_layers
            p["ecg_channels"] = ecg_channels
            p["skna_channels"] = skna_channels
            p["iskna_rskna_channels"] = iskna_rskna_channels

            use_fd = p.get("use_fd", False)
            use_fft = p.get("use_fft", False)
            use_psd = p.get("use_psd", False)
            use_cwt = p.get("use_cwt", False)
            fd_channels = tuple(p.get("fd_channels", cnn_channels))
            fft_channels = tuple(p.get("fft_channels", cnn_channels))
            psd_channels = tuple(p.get("psd_channels", cnn_channels))
            cwt_channels = tuple(p.get("cwt_channels", (8, 16, 32)))

            p["use_fd"] = use_fd
            p["use_fft"] = use_fft
            p["use_psd"] = use_psd
            p["use_cwt"] = use_cwt
            p["fd_channels"] = fd_channels
            p["fft_channels"] = fft_channels
            p["psd_channels"] = psd_channels
            p["cwt_channels"] = cwt_channels

            # Both the legacy 2-channel FD branch and the split FFT/PSD branches
            # read the same 2-channel fd_signal.npz, so load it if any are on.
            need_fd = use_fd or use_fft or use_psd

            # Individually switchable CNN input branches (for ECG-only / SKNA-only
            # style ablations). Default on so older param dicts behave as before.
            use_ecg = p.get("use_ecg", True)
            use_skna = p.get("use_skna", True)
            use_iskna_rskna = p.get("use_iskna_rskna", True)
            p["use_ecg"] = use_ecg
            p["use_skna"] = use_skna
            p["use_iskna_rskna"] = use_iskna_rskna
            
            cnn_checkpoint_above = p.get("cnn_checkpoint_above", CNN_CHECKPOINT_ABOVE)
            p["cnn_checkpoint_above"] = cnn_checkpoint_above

            dropout = p.get("dropout", 0.3)
            weight_decay = p.get("weight_decay", 1e-4)
            anti_collapse_lambda = p.get("anti_collapse_lambda", 0.0)
            p["anti_collapse_lambda"] = anti_collapse_lambda  
            use_askna = p.get("use_askna", True)
            p["use_askna"] = use_askna
            active_inputs = [name for name, on in [
                ("ECG", use_ecg), ("filtered-SKNA", use_skna),
                ("iSKNA+rSKNA", use_iskna_rskna), ("aSKNA", use_askna),
                ("FD", use_fd), ("FFT", use_fft), ("PSD", use_psd),
                ("CWT", use_cwt)] if on]
            self.log.emit(f"CNN inputs: {', '.join(active_inputs) or '(none!)'}")
            self.log.emit(
                f"1D branch architecture: {cnn_arch}"
                + (" (canonical ResNet-18: 4 stages of [2,2,2,2] BasicBlocks at widths "
                   "(64,128,256,512) per waveform branch. Fixed widths override the "
                   "*_channels args. FD/FFT/PSD stay plain. Memory bounded by gradient "
                   "checkpointing.)"
                   if cnn_arch == "resnet18" else
                   f" (residual 1D ResNet for the ECG/SKNA/iSKNA+rSKNA waveforms - strided "
                   f"stem + {resnet_blocks_per_stage} residual block(s)/stage at the "
                   f"*_channels widths; adds trainable depth, memory stays bounded by "
                   f"gradient checkpointing. FD/FFT/PSD stay plain.)"
                   if cnn_arch == "resnet" else
                   f" (fully-connected MLP, no convolution: each window is "
                   f"adaptive-avg-pooled to {ann_pool_len} samples, flattened, then passed "
                   f"through Linear->BN->ReLU->Dropout at the *_channels widths. Applies to "
                   f"every 1D branch including FD/FFT/PSD.)"
                   if cnn_arch == "ann" else
                   " (original stacked Conv->BN->ReLU->MaxPool blocks)"))
            if use_cwt:
                self.log.emit(
                    f"2D (CWT scalogram) branch architecture: {cwt_arch}"
                    + (" (2D ResNet-18: 4 stages of [2,2,2,2] BasicBlocks at widths "
                       "(64,128,256,512) over the scalogram; fixed widths override "
                       "cwt_channels)"
                       if cwt_arch == "resnet18" else
                       f" (2D ResNet: strided stem + {resnet_blocks_per_stage} residual "
                       f"block(s)/stage at the cwt_channels widths)"
                       if cwt_arch == "resnet" else
                       " (original stacked Conv2d->BN->ReLU->MaxPool blocks)"))
            # Per-subject fine-tuning (personalization / hybrid calibration). When
            # on, each fold additionally splits the held-out TEST subject's own
            # recording by time into adapt/val/test, freezes all but the last LSTM
            # layer + attention + head, fine-tunes on the adapt slice, and reports
            # base-vs-personalized MAE + delta-correlation on the held-out test
            # slice. Default off -> existing LOSO behaviour is untouched. NOTE: the
            # personalized metrics are on that smaller held-out slice, NOT on the
            # whole subject, so they are not directly comparable to the plain LOSO
            # MAE columns - they are only comparable to p_base_* (same slice).
            personalize = bool(p.get("personalize", False))
            ft_lr = p.get("ft_lr", 2e-4)
            ft_epochs = int(p.get("ft_epochs", 40))
            ft_patience = int(p.get("ft_patience", 8))
            adapt_end = float(p.get("adapt_end", 0.5))
            val_end = float(p.get("val_end", 0.6))
            p["personalize"] = personalize
            p["ft_lr"] = ft_lr
            p["ft_epochs"] = ft_epochs
            p["ft_patience"] = ft_patience
            p["adapt_end"] = adapt_end
            p["val_end"] = val_end
            if personalize:
                self.log.emit(
                    f"Personalization: ON (per-subject fine-tune of last LSTM layer + "
                    f"attention + head; adapt<{adapt_end:.0%}, val {adapt_end:.0%}-{val_end:.0%}, "
                    f"test >{val_end:.0%} of each held-out subject's recording; ft_lr={ft_lr}, "
                    f"ft_epochs={ft_epochs}, ft_patience={ft_patience}). Personalized metrics are "
                    f"on the held-out test slice, comparable to p_base_* only.")

            calib_windows = p.get("calib_windows", 10)
            norm_mode = p.get("norm_mode", "global")
            p["norm_mode"] = norm_mode
            exclude_subjects = [str(s).strip() for s in p.get("exclude_subjects", []) if str(s).strip()]
            p["exclude_subjects"] = exclude_subjects
            device = "cuda" if (p.get("use_gpu", True) and torch.cuda.is_available()) else "cpu"

            self.log.emit(f"Device: {device}")
            self.log.emit(f"Loss: Huber (SmoothL1)"
                           + (f" + anti-collapse penalty, anti_collapse_lambda={anti_collapse_lambda}"
                              if anti_collapse_lambda > 0 else " (anti-collapse penalty off)"))
            self.log.emit(f"Input norm: {norm_mode}"
                           + (" (each recording z-scored by its own calibration windows)"
                              if norm_mode == "calib" else " (single train-set mean/std for all subjects)"))
            windows_per_batch = batch_size * seq_len
            if cnn_checkpoint_above and windows_per_batch > cnn_checkpoint_above:
                self.log.emit(
                    f"CNN branches: {windows_per_batch} windows/batch (batch_size={batch_size} x "
                    f"seq_len={seq_len}) exceeds cnn_checkpoint_above={cnn_checkpoint_above}, so "
                    f"they will run with gradient checkpointing - same results, bounded memory, "
                    f"~50% slower per epoch. Lower batch_size or seq_len to avoid this; set "
                    f"cnn_checkpoint_above=0 to disable it (risks the process being OOM-killed).")
            self.log.emit(f"Loading aggregated dataset from {combined_dir} "
                           f"(seq_len={seq_len}, stride={stride}, calib_windows={calib_windows})...")
            if downsample > 1:
                self.log.emit(
                    f"Downsampling time-domain signals (ECG/SKNA/iSKNA/rSKNA) by "
                    f"{downsample}x at load (anti-aliased): 10 kHz -> {10000 // downsample} Hz, "
                    f"50000 -> {50000 // downsample} samples/window. Cuts input RAM and CNN "
                    f"activation memory ~{downsample}x. aSKNA/FD/CWT are left at full resolution.")

            fd_npz = cwt_npz = None
            if need_fd:
                fd_npz = os.path.join(combined_dir, "all_fd_signal.npz")
                if not os.path.isfile(fd_npz):
                    raise FileNotFoundError(
                        f"use_fd/use_fft/use_psd=True but {fd_npz} not found. Enable 'Compute FD "
                        "arrays' in the preprocessing pipeline for every recording, then re-run "
                        "aggregate_datasets.py.")
            if use_cwt:
                cwt_npz = os.path.join(combined_dir, "all_cwt_signal.npz")
                if not os.path.isfile(cwt_npz):
                    raise FileNotFoundError(
                        f"use_cwt=True but {cwt_npz} not found. Enable 'Compute CWT arrays' in the "
                        "preprocessing pipeline for every recording, then re-run aggregate_datasets.py.")

            ds = SKNASequenceDataset(
                raw_signal_npz=os.path.join(combined_dir, "all_raw_signal.npz"),
                raw_signal_labels_csv=os.path.join(combined_dir, "all_raw_signal_labels.csv"),
                skna_signal_npz=os.path.join(combined_dir, "all_skna_signal.npz"),
                iskna_rskna_npz=os.path.join(combined_dir, "all_iskna_rskna.npz"),
                askna_npz=os.path.join(combined_dir, "all_askna.npz"),
                fd_signal_npz=fd_npz, cwt_signal_npz=cwt_npz,
                seq_len=seq_len, stride=stride, calib_windows=calib_windows,
                norm_mode=norm_mode, downsample=downsample,
            )
            groups = ds.sequence_groups()
            self.log.emit(f"Loaded {len(ds)} sequences across "
                           f"{len(set(groups))} subjects: {sorted(set(groups))}")

            if exclude_subjects:
                unknown = sorted(set(exclude_subjects) - set(groups))
                if unknown:
                    raise ValueError(
                        f"exclude_subjects lists {unknown}, which are not in the dataset. "
                        f"Available subjects: {sorted(set(groups))}")
                keep = np.where(~np.isin(groups, exclude_subjects))[0]
                dropped = len(ds) - len(keep)
                ds = ds.subset(keep)
                groups = ds.sequence_groups()
                self.log.emit(f"Excluded subjects {sorted(exclude_subjects)}: dropped {dropped} "
                               f"sequences, {len(ds)} remain across {len(set(groups))} subjects.")

            subjects = sorted(set(groups))

            if len(subjects) < 3:
                raise ValueError(
                    "Need at least 3 subjects for a LOSO train/val/test split "
                    "(1 test + 1 val + >=1 train). Only found: " + str(subjects))

            val_frac = p.get("val_frac", 0.15)
            val_mode = p.get("val_mode", "time_split")  # conservative fallback for old saved params
            p["val_mode"] = val_mode
            if val_mode not in ("subject_holdout", "time_split"):
                raise ValueError(f"Unknown val_mode {val_mode!r}, expected 'subject_holdout' or 'time_split'.")

            fold_results = {}
            for fold_i, test_subject in enumerate(subjects):
                if self._should_stop():
                    self.log.emit("Training cancelled.")
                    return

                remaining = [s for s in subjects if s != test_subject]

                fold_val_mode = val_mode
                if fold_val_mode == "subject_holdout" and len(remaining) < 2:
                    self.log.emit(f"  [val_mode] only {len(remaining)} non-test subject(s) available - "
                                   f"can't hold one out for validation too, falling back to time_split "
                                   f"for this fold.")
                    fold_val_mode = "time_split"

                self.fold_started.emit(test_subject)

                if fold_val_mode == "subject_holdout":
                    sorted_remaining = sorted(remaining)
                    val_subject = sorted_remaining[fold_i % len(sorted_remaining)]
                    val_mask = groups == val_subject
                    train_mask = np.isin(groups, remaining) & ~val_mask
                    train_seq_idx = np.where(train_mask)[0]
                    val_seq_idx = np.where(val_mask)[0]
                    actual_train_subjects = [s for s in remaining if s != val_subject]
                    self.log.emit(f"\n=== Fold {fold_i + 1}/{len(subjects)}: test={test_subject}, "
                                   f"val={val_subject} (whole subject held out, nested LOSO), "
                                   f"train={actual_train_subjects} ===")
                else:
                    train_seq_idx, val_seq_idx = ds.split_within_subjects(remaining, val_frac=val_frac)
                    actual_train_subjects = remaining
                    val_subject = None
                    self.log.emit(f"\n=== Fold {fold_i + 1}/{len(subjects)}: test={test_subject}, "
                                   f"train+val pool={remaining} (val_frac={val_frac}, split by time "
                                   f"within each subject's recording, not a held-out subject) ===")

                if len(val_seq_idx) == 0:
                    detail = (
                        f"A sequence only counts as validation if all {seq_len} of its windows fall "
                        f"in the last val_frac={val_frac} of a recording, and no recording in "
                        f"{remaining} is long enough for that. Reduce seq_len, raise val_frac, or "
                        f"use val_mode='subject_holdout'."
                        if fold_val_mode == "time_split" else
                        f"The held-out validation subject {val_subject} has no sequences of "
                        f"seq_len={seq_len} - its recording is shorter than that.")
                    raise ValueError(f"Fold test={test_subject}: no validation sequences. {detail}")
                if len(train_seq_idx) == 0:
                    raise ValueError(
                        f"Fold test={test_subject}: no training sequences left after the "
                        f"train/val time split (seq_len={seq_len}, val_frac={val_frac}).")

                train_ds = ds.subset(train_seq_idx, fit_norm=True)
                val_ds = ds.subset(val_seq_idx, norm_stats=train_ds.norm_stats)
                test_ds = ds.subset(np.where(groups == test_subject)[0], norm_stats=train_ds.norm_stats)
                self.log.emit(f"  sequences -> train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

                train_labels = train_ds._arrays["labels"]
                train_pos = train_ds.seq_index[:, -1]
                train_targets_abs = torch.tensor(
                    train_labels[["SBP", "DBP"]].iloc[train_pos].to_numpy(), dtype=torch.float32)
                train_baseline = torch.tensor(np.stack([
                    train_ds._arrays["baseline_sbp"][train_pos],
                    train_ds._arrays["baseline_dbp"][train_pos],
                ], axis=1), dtype=torch.float32)
                train_targets_delta = train_targets_abs - train_baseline
                target_scaler = TargetScaler().fit(train_targets_delta)

                train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
                test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

                model = SKNABPModel(
                    ecg_channels=ecg_channels,
                    skna_channels=skna_channels,
                    iskna_rskna_channels=iskna_rskna_channels,
                    lstm_hidden=lstm_hidden, lstm_layers=lstm_layers,
                    dropout=dropout,
                    use_ecg=use_ecg, use_skna=use_skna, use_iskna_rskna=use_iskna_rskna,
                    use_askna=use_askna,
                    use_fd=use_fd, fd_channels=fd_channels,
                    use_fft=use_fft, fft_channels=fft_channels,
                    use_psd=use_psd, psd_channels=psd_channels,
                    use_cwt=use_cwt, cwt_channels=cwt_channels,
                    cnn_checkpoint_above=cnn_checkpoint_above,
                    cnn_arch=cnn_arch,
                    cwt_arch=cwt_arch,
                    resnet_blocks_per_stage=resnet_blocks_per_stage,
                    ann_pool_len=ann_pool_len,
                ).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', factor=0.5, patience=max(patience // 2, 2))
                stopper = EarlyStopper(patience=patience, min_delta=0.01)
                best_state, best_epoch = None, -1

                for epoch in range(epochs):
                    if self._should_stop():
                        self.log.emit("Training cancelled mid-fold.")
                        return
                    train_loss = train_one_epoch(model, train_loader, optimizer, device, target_scaler,
                                                  anti_collapse_lambda)
                    val_metrics = evaluate(model, val_loader, device, target_scaler,
                                            anti_collapse_lambda=anti_collapse_lambda)
                    val_loss = val_metrics["loss"]
                    val_select = val_metrics["MAE_SBP"] + val_metrics["MAE_DBP"]
                    scheduler.step(val_select)
                    self.epoch_progress.emit(test_subject, epoch, train_loss, val_loss, val_select)

                    improved = stopper.step(val_select)
                    if improved:
                        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                        best_epoch = epoch
                    if improved or epoch % 5 == 0:
                        self.log.emit(
                            f"  epoch {epoch}: train_loss={train_loss:.4f} "
                            f"val_MAE_SBP={val_metrics['MAE_SBP']:.2f} val_MAE_DBP={val_metrics['MAE_DBP']:.2f} "
                            f"(select={val_select:.3f} mmHg; val_loss={val_loss:.4f}, normalized - "
                            f"not comparable across folds)"
                            f"{'  * best' if improved else ''}")
                    if stopper.should_stop:
                        self.log.emit(f"  Early stopping at epoch {epoch} (best epoch {best_epoch}).")
                        break

                if best_state is not None:
                    model.load_state_dict(best_state)

                test_metrics = evaluate(model, test_loader, device, target_scaler, return_predictions=True,
                                         anti_collapse_lambda=anti_collapse_lambda)
                self.log.emit(f"  TEST ({test_subject}): MAE_SBP={test_metrics['MAE_SBP']:.2f}  "
                               f"MAE_DBP={test_metrics['MAE_DBP']:.2f}  (best_epoch={best_epoch})")
                self.log.emit(
                    f"  ME +/- SD (mmHg): SBP {test_metrics['bias_SBP']:+.2f} +/- {test_metrics['SDE_SBP']:.2f}  "
                    f"DBP {test_metrics['bias_DBP']:+.2f} +/- {test_metrics['SDE_DBP']:.2f}  "
                    f"(ME = mean signed error, SD = its spread; MAE hides error direction, this shows it)")
                self.log.emit(f"  corr_SBP_delta={test_metrics['corr_SBP_delta']:.3f}  "
                               f"corr_DBP_delta={test_metrics['corr_DBP_delta']:.3f}  "
                               f"(tracking correlation on change-from-calibration, independent "
                               f"of the calibration offset - the trustworthy correlation number. "
                               f"corr_SBP={test_metrics['corr_SBP']:.3f}/corr_DBP={test_metrics['corr_DBP']:.3f} "
                               f"in absolute-mmHg space can look inflated for multi-session subjects "
                               f"whose sessions sit at different BP offsets - don't read those as skill.)")
                self.log.emit(f"  pred_std_SBP={test_metrics['pred_std_SBP']:.2f} "
                               f"(true_std_SBP={test_metrics['true_std_SBP']:.2f})  "
                               f"pred_std_DBP={test_metrics['pred_std_DBP']:.2f} "
                               f"(true_std_DBP={test_metrics['true_std_DBP']:.2f})"
                               + ("  <- WARNING: model barely varies its output, likely "
                                  "predicting near a constant" if test_metrics['pred_std_SBP']
                                  < 0.3 * test_metrics['true_std_SBP'] else ""))

                self.log.emit(
                    f"  bias_SBP={test_metrics['bias_SBP']:+.2f}  bias_DBP={test_metrics['bias_DBP']:+.2f}  "
                    f"MAE_SBP_debiased={test_metrics['MAE_SBP_debiased']:.2f}  "
                    f"MAE_DBP_debiased={test_metrics['MAE_DBP_debiased']:.2f}  "
                    f"MAE_SBP_oracle_const={test_metrics['MAE_SBP_oracle_const']:.2f}  "
                    f"MAE_DBP_oracle_const={test_metrics['MAE_DBP_oracle_const']:.2f}"
                    + ("" if (test_metrics['has_tracking_skill_SBP'] and test_metrics['has_tracking_skill_DBP'])
                       else "  <- WARNING: no tracking skill beyond predicting this subject's own mean BP"))

                # Baseline (zero-delta / calibration-only): predict "no change
                # from the held-out subject's OWN calibration reading" - i.e.
                # pred_delta=0 for every window. This uses per-subject info
                # (the test subject's own baseline), exactly what the
                # delta-framed model is also given, so THIS is the real bar
                # the model needs to clear to prove it's learning actual BP
                # dynamics from ECG/SKNA rather than just inheriting the
                # calibration reading.
                baseline_calib_mae_sbp = float(np.abs(test_metrics["true_delta"][:, 0]).mean())
                baseline_calib_mae_dbp = float(np.abs(test_metrics["true_delta"][:, 1]).mean())
                beats_baseline_calib = (test_metrics["MAE_SBP_delta"] < baseline_calib_mae_sbp
                                         and test_metrics["MAE_DBP_delta"] < baseline_calib_mae_dbp)
                self.log.emit(
                    f"  BASELINE/zero-delta [the real bar] (always predict the subject's own "
                    f"calibration reading, i.e. no BP change): MAE_SBP={baseline_calib_mae_sbp:.2f}  "
                    f"MAE_DBP={baseline_calib_mae_dbp:.2f}"
                    + ("" if beats_baseline_calib else
                       "  <- WARNING: model does not beat 'assume no change from calibration'"))

                # Baseline (population mean, context only): predict the TRAIN
                # pool's mean SBP/DBP for every window of the held-out test
                # subject - no signal used, no per-subject calibration either.
                # This is a weaker/different comparison than zero-delta (it
                # doesn't get the subject's own baseline reading), kept only
                # for continuity with earlier runs - the zero-delta baseline
                # above is the one that matters.
                train_mean_sbp = train_targets_abs[:, 0].mean().item()
                train_mean_dbp = train_targets_abs[:, 1].mean().item()
                baseline_mae_sbp = float(np.abs(test_metrics["true"][:, 0] - train_mean_sbp).mean())
                baseline_mae_dbp = float(np.abs(test_metrics["true"][:, 1] - train_mean_dbp).mean())
                beats_baseline = (test_metrics["MAE_SBP"] < baseline_mae_sbp
                                   and test_metrics["MAE_DBP"] < baseline_mae_dbp)
                self.log.emit(
                    f"  BASELINE/pop-mean [context only] (always predict train-mean "
                    f"SBP={train_mean_sbp:.1f}, DBP={train_mean_dbp:.1f}): "
                    f"MAE_SBP={baseline_mae_sbp:.2f}  MAE_DBP={baseline_mae_dbp:.2f}"
                    + ("" if beats_baseline else
                       "  <- WARNING: model does not beat this constant-output baseline"))

                fold_dir = os.path.join(out_dir, f"fold_{test_subject}")
                os.makedirs(fold_dir, exist_ok=True)
                torch.save({
                    "model_state": model.state_dict(),
                    "target_scaler": target_scaler.state_dict(),
                    "norm_stats": train_ds.norm_stats,
                    "params": p,
                    # Recorded so InferenceWorker can refuse/warn if asked to evaluate
                    # this checkpoint on a subject it already saw during training -
                    # see the leakage guard in inference_worker.py. train_subjects is
                    # the set that actually fed the optimizer (excludes val_subject
                    # under subject_holdout mode, since that subject's gradients were
                    # never used - only its loss was read for early stopping).
                    "test_subject": test_subject,
                    "train_subjects": actual_train_subjects,
                    "val_mode": fold_val_mode,
                    "val_subject": val_subject,
                }, os.path.join(fold_dir, "checkpoint.pt"))

                pred_df = pd.DataFrame({
                    "Subject_ID": test_metrics["subject_id"],
                    "SBP_true": test_metrics["true"][:, 0], "SBP_pred": test_metrics["pred"][:, 0],
                    "DBP_true": test_metrics["true"][:, 1], "DBP_pred": test_metrics["pred"][:, 1],
                })
                pred_df.to_csv(os.path.join(fold_dir, "predictions.csv"), index=False)

                # ---- Per-subject fine-tuning (personalization) ----
                # The base checkpoint above already holds the population weights,
                # so we can fine-tune `model` in place here without corrupting it.
                personal = {}
                if personalize:
                    adapt_idx, pval_idx, ptest_idx = three_way_time_split(
                        ds, test_subject, adapt_end=adapt_end, val_end=val_end)
                    if len(adapt_idx) == 0 or len(pval_idx) == 0 or len(ptest_idx) == 0:
                        self.log.emit(
                            f"  [personalize] skipped {test_subject}: too few windows to split "
                            f"(adapt={len(adapt_idx)}, val={len(pval_idx)}, test={len(ptest_idx)}).")
                    else:
                        adapt_ds = ds.subset(adapt_idx, norm_stats=train_ds.norm_stats)
                        pval_ds = ds.subset(pval_idx, norm_stats=train_ds.norm_stats)
                        ptest_ds = ds.subset(ptest_idx, norm_stats=train_ds.norm_stats)
                        adapt_loader = DataLoader(adapt_ds, batch_size=batch_size, shuffle=True)
                        pval_loader = DataLoader(pval_ds, batch_size=batch_size, shuffle=False)
                        ptest_loader = DataLoader(ptest_ds, batch_size=batch_size, shuffle=False)

                        # Base (un-personalized) reference on the held-out slice.
                        base_slice = evaluate(model, ptest_loader, device, target_scaler,
                                               return_predictions=True,
                                               anti_collapse_lambda=anti_collapse_lambda)

                        # Freeze all but last LSTM layer + attention + head, then fine-tune.
                        freeze_for_hybrid_calibration(model)
                        ft_params = [q for q in model.parameters() if q.requires_grad]
                        ft_opt = torch.optim.Adam(ft_params, lr=ft_lr, weight_decay=weight_decay)
                        ft_stopper = EarlyStopper(patience=ft_patience, min_delta=0.01)
                        best_ft_state, best_ft_epoch = None, -1
                        for ft_epoch in range(ft_epochs):
                            if self._should_stop():
                                self.log.emit("Training cancelled during personalization.")
                                return
                            train_one_epoch(model, adapt_loader, ft_opt, device, target_scaler,
                                             anti_collapse_lambda)
                            vm = evaluate(model, pval_loader, device, target_scaler,
                                           anti_collapse_lambda=anti_collapse_lambda)
                            if ft_stopper.step(vm["MAE_SBP"] + vm["MAE_DBP"]):
                                best_ft_state = {k: v.detach().cpu().clone()
                                                 for k, v in model.state_dict().items()}
                                best_ft_epoch = ft_epoch
                            if ft_stopper.should_stop:
                                break
                        if best_ft_state is not None:
                            model.load_state_dict(best_ft_state)

                        post_slice = evaluate(model, ptest_loader, device, target_scaler,
                                               return_predictions=True,
                                               anti_collapse_lambda=anti_collapse_lambda)

                        # Absolute-BP predictions of the PERSONALIZED model on its
                        # held-out test slice, same columns as the base predictions
                        # frame, so they can feed the results table and the BP
                        # device-validation standards (BHS/AAMI/ISO/IEEE) computed
                        # on the tuned model exactly as the base ones are.
                        p_pred_df = pd.DataFrame({
                            "Subject_ID": post_slice["subject_id"],
                            "SBP_true": post_slice["true"][:, 0], "SBP_pred": post_slice["pred"][:, 0],
                            "DBP_true": post_slice["true"][:, 1], "DBP_pred": post_slice["pred"][:, 1],
                        })
                        p_pred_df.to_csv(os.path.join(fold_dir, "predictions_personalized.csv"), index=False)

                        # Zero-delta baseline on the SAME slice: predict no change
                        # from calibration (mean |true_delta|) - the real bar.
                        p_bl_sbp = float(np.abs(post_slice["true_delta"][:, 0]).mean())
                        p_bl_dbp = float(np.abs(post_slice["true_delta"][:, 1]).mean())
                        p_beats = bool(post_slice["MAE_SBP"] < p_bl_sbp
                                        and post_slice["MAE_DBP"] < p_bl_dbp)

                        # Restore the population weights so the saved base checkpoint
                        # and any later reuse of `model` are unaffected; also save a
                        # separate personalized checkpoint for reference.
                        torch.save({"model_state": model.state_dict(),
                                     "target_scaler": target_scaler.state_dict(),
                                     "norm_stats": train_ds.norm_stats, "params": p,
                                     "test_subject": test_subject,
                                     "train_subjects": actual_train_subjects,
                                     "personalized": True},
                                    os.path.join(fold_dir, "checkpoint_personalized.pt"))
                        model.load_state_dict(best_state)

                        personal = {
                            "p_n_test": len(ptest_ds),
                            "p_base_MAE_SBP": base_slice["MAE_SBP"], "p_base_MAE_DBP": base_slice["MAE_DBP"],
                            "p_post_MAE_SBP": post_slice["MAE_SBP"], "p_post_MAE_DBP": post_slice["MAE_DBP"],
                            "p_baseline_MAE_SBP": p_bl_sbp, "p_baseline_MAE_DBP": p_bl_dbp,
                            "p_corr_SBP_delta": post_slice["corr_SBP_delta"],
                            "p_corr_DBP_delta": post_slice["corr_DBP_delta"],
                            "p_beats_baseline": p_beats, "p_best_epoch": best_ft_epoch,
                            # Tuned-model predictions on the held-out slice, pooled
                            # across folds for the personalized results table + the
                            # personalized BP device-validation standards.
                            "p_predictions": p_pred_df,
                        }
                        self.log.emit(
                            f"  PERSONALIZED ({test_subject}) on held-out slice (n={len(ptest_ds)}): "
                            f"base MAE {base_slice['MAE_SBP']:.2f}/{base_slice['MAE_DBP']:.2f} -> "
                            f"post {post_slice['MAE_SBP']:.2f}/{post_slice['MAE_DBP']:.2f}  "
                            f"(zero-delta baseline {p_bl_sbp:.2f}/{p_bl_dbp:.2f}; "
                            f"corr_delta {post_slice['corr_SBP_delta']:.2f}/{post_slice['corr_DBP_delta']:.2f}; "
                            f"beats baseline: {p_beats})")

                fold_result = {
                    "test_subject": test_subject, "train_val_pool": remaining,
                    "val_mode": fold_val_mode, "val_subject": val_subject,
                    "val_frac": val_frac, "best_epoch": best_epoch,
                    "MAE_SBP": test_metrics["MAE_SBP"], "MAE_DBP": test_metrics["MAE_DBP"],
                    "corr_SBP": test_metrics["corr_SBP"], "corr_DBP": test_metrics["corr_DBP"],
                    "corr_SBP_delta": test_metrics["corr_SBP_delta"], "corr_DBP_delta": test_metrics["corr_DBP_delta"],
                    "bias_SBP": test_metrics["bias_SBP"], "bias_DBP": test_metrics["bias_DBP"],
                    "SDE_SBP": test_metrics["SDE_SBP"], "SDE_DBP": test_metrics["SDE_DBP"],
                    "MAE_SBP_debiased": test_metrics["MAE_SBP_debiased"],
                    "MAE_DBP_debiased": test_metrics["MAE_DBP_debiased"],
                    "MAE_SBP_oracle_const": test_metrics["MAE_SBP_oracle_const"],
                    "MAE_DBP_oracle_const": test_metrics["MAE_DBP_oracle_const"],
                    "has_tracking_skill_SBP": test_metrics["has_tracking_skill_SBP"],
                    "has_tracking_skill_DBP": test_metrics["has_tracking_skill_DBP"],
                    "MAE_SBP_baseline_calib": baseline_calib_mae_sbp,
                    "MAE_DBP_baseline_calib": baseline_calib_mae_dbp,
                    "beats_baseline_calib": beats_baseline_calib,
                    "MAE_SBP_baseline": baseline_mae_sbp, "MAE_DBP_baseline": baseline_mae_dbp,
                    "beats_baseline": beats_baseline,
                    "n_test_seq": len(test_ds), "predictions": pred_df,
                }
                fold_result.update(personal)
                fold_results[test_subject] = fold_result
                self.fold_finished.emit(test_subject, fold_result)

            summary_rows = []
            for k, v in fold_results.items():
                row = {"Subject_ID": k, "val_mode": v["val_mode"], "val_subject": v["val_subject"],
                       "MAE_SBP": v["MAE_SBP"], "MAE_DBP": v["MAE_DBP"],
                       "MAE_SBP_baseline_calib": v["MAE_SBP_baseline_calib"],
                       "MAE_DBP_baseline_calib": v["MAE_DBP_baseline_calib"],
                       "beats_baseline_calib": v["beats_baseline_calib"],
                       "bias_SBP": v["bias_SBP"], "bias_DBP": v["bias_DBP"],
                       "SDE_SBP": v["SDE_SBP"], "SDE_DBP": v["SDE_DBP"],
                       "MAE_SBP_debiased": v["MAE_SBP_debiased"], "MAE_DBP_debiased": v["MAE_DBP_debiased"],
                       "MAE_SBP_oracle_const": v["MAE_SBP_oracle_const"],
                       "MAE_DBP_oracle_const": v["MAE_DBP_oracle_const"],
                       "has_tracking_skill_SBP": v["has_tracking_skill_SBP"],
                       "has_tracking_skill_DBP": v["has_tracking_skill_DBP"],
                       "corr_SBP_delta": v["corr_SBP_delta"], "corr_DBP_delta": v["corr_DBP_delta"],
                       "corr_SBP": v["corr_SBP"], "corr_DBP": v["corr_DBP"],
                       "MAE_SBP_baseline": v["MAE_SBP_baseline"], "MAE_DBP_baseline": v["MAE_DBP_baseline"],
                       "beats_baseline": v["beats_baseline"],
                       "best_epoch": v["best_epoch"], "n_test_seq": v["n_test_seq"]}
                if personalize:
                    # Personalized metrics live on the smaller held-out slice; compare
                    # to p_base_* (same slice), NOT to the plain-LOSO MAE columns.
                    for key in ("p_n_test", "p_base_MAE_SBP", "p_base_MAE_DBP",
                                "p_post_MAE_SBP", "p_post_MAE_DBP",
                                "p_baseline_MAE_SBP", "p_baseline_MAE_DBP",
                                "p_corr_SBP_delta", "p_corr_DBP_delta",
                                "p_beats_baseline", "p_best_epoch"):
                        row[key] = v.get(key)
                summary_rows.append(row)
            summary = pd.DataFrame(summary_rows)
            summary.to_csv(os.path.join(out_dir, "loso_summary.csv"), index=False)
            self.log.emit("\n=== LOSO complete ===")
            self.log.emit(summary.to_string(index=False))

            if personalize and "p_post_MAE_SBP" in summary:
                ps = summary.dropna(subset=["p_post_MAE_SBP"])
                if len(ps):
                    n_beats = int(ps["p_beats_baseline"].sum())
                    n_pos = int((ps["p_corr_SBP_delta"] > 0).sum())
                    self.log.emit(
                        f"\n=== Personalization (held-out slice) ===\n"
                        f"  base MAE  SBP {ps['p_base_MAE_SBP'].mean():.2f}  DBP {ps['p_base_MAE_DBP'].mean():.2f}\n"
                        f"  post MAE  SBP {ps['p_post_MAE_SBP'].mean():.2f}  DBP {ps['p_post_MAE_DBP'].mean():.2f}\n"
                        f"  zero-delta baseline  SBP {ps['p_baseline_MAE_SBP'].mean():.2f}  "
                        f"DBP {ps['p_baseline_MAE_DBP'].mean():.2f}\n"
                        f"  folds where personalized beats zero-delta baseline: {n_beats}/{len(ps)}\n"
                        f"  folds with positive SBP delta-correlation (real tracking): {n_pos}/{len(ps)}  "
                        f"(mean corr_delta SBP {ps['p_corr_SBP_delta'].mean():.2f}) "
                        f"<- the metric that separates tracking from offset-only gains")

            # Device-validation standards (BHS / AAMI SP10 / ISO 81060-2:2018 /
            # IEEE 1708) on the POOLED cross-subject predictions. These grade the
            # absolute-BP error distribution and need many subjects, so they only
            # make sense on the concatenation of every fold's held-out readings -
            # never per fold. Reported next to the zero-delta baseline so a
            # grade is never read without the bar the model has to clear.
            bp_standards = None
            try:
                from bp_standards import (compute_bp_standards_from_df,
                                          format_standards_report,
                                          standards_summary_rows)
                pooled_pred = pd.concat(
                    [v["predictions"] for v in fold_results.values()],
                    axis=0, ignore_index=True)
                bp_standards = compute_bp_standards_from_df(pooled_pred)
                zero_delta_mae = {
                    "SBP": float(summary["MAE_SBP_baseline_calib"].mean()),
                    "DBP": float(summary["MAE_DBP_baseline_calib"].mean()),
                }
                report = format_standards_report(bp_standards, zero_delta_mae)
                self.log.emit("\n" + report)
                with open(os.path.join(out_dir, "bp_standards.txt"), "w") as f:
                    f.write(report + "\n")
                pd.DataFrame(standards_summary_rows(bp_standards)).to_csv(
                    os.path.join(out_dir, "bp_standards.csv"), index=False)
            except Exception as e:
                self.log.emit(f"\n[bp_standards] Could not compute device-validation "
                               f"standards: {e}")

            # Same standards, but on the PERSONALIZED (fine-tuned) predictions.
            # These live on each subject's smaller held-out test slice, so the
            # pool is smaller than the base pool and the grades are only
            # comparable to a base-model pool restricted to the same slices -
            # NOT to the whole-subject base standards above. Reported next to the
            # personalized zero-delta baseline (p_baseline_*), the real bar.
            bp_standards_personalized = None
            if personalize:
                try:
                    from bp_standards import (compute_bp_standards_from_df,
                                              format_standards_report,
                                              standards_summary_rows)
                    p_frames = [v["p_predictions"] for v in fold_results.values()
                                if v.get("p_predictions") is not None]
                    if not p_frames:
                        self.log.emit("\n[bp_standards] Personalization on, but no fold "
                                       "produced a personalized test slice - skipping tuned standards.")
                    else:
                        pooled_p = pd.concat(p_frames, axis=0, ignore_index=True)
                        bp_standards_personalized = compute_bp_standards_from_df(pooled_p)
                        p_summary = summary.dropna(subset=["p_baseline_MAE_SBP"]) \
                            if "p_baseline_MAE_SBP" in summary else summary.iloc[0:0]
                        p_zero_delta = None
                        if len(p_summary):
                            p_zero_delta = {
                                "SBP": float(p_summary["p_baseline_MAE_SBP"].mean()),
                                "DBP": float(p_summary["p_baseline_MAE_DBP"].mean()),
                            }
                        p_report = format_standards_report(bp_standards_personalized, p_zero_delta)
                        self.log.emit(
                            "\n### PERSONALIZED (fine-tuned) model - held-out test slices only ###\n"
                            + p_report)
                        with open(os.path.join(out_dir, "bp_standards_personalized.txt"), "w") as f:
                            f.write(p_report + "\n")
                        pd.DataFrame(standards_summary_rows(bp_standards_personalized)).to_csv(
                            os.path.join(out_dir, "bp_standards_personalized.csv"), index=False)
                except Exception as e:
                    self.log.emit(f"\n[bp_standards] Could not compute personalized "
                                   f"device-validation standards: {e}")

            n_beats_calib = int(summary["beats_baseline_calib"].sum())
            n_skill_sbp = int(summary["has_tracking_skill_SBP"].sum())
            n_skill_dbp = int(summary["has_tracking_skill_DBP"].sum())
            self.log.emit(
                f"\nFolds beating the zero-delta/calibration-only baseline: "
                f"{n_beats_calib}/{len(summary)}  <- THE REAL BAR TO CLEAR")
            self.log.emit(
                f"Folds with real tracking skill (MAE_debiased < MAE_oracle_const): "
                f"SBP {n_skill_sbp}/{len(summary)}  DBP {n_skill_dbp}/{len(summary)}  "
                f"<- if this is low, the model is mostly guessing each subject's own "
                f"average BP, not tracking anything from ECG/SKNA")
            self.log.emit(
                f"Mean MAE_SBP={summary['MAE_SBP'].mean():.2f} "
                f"(zero-delta baseline={summary['MAE_SBP_baseline_calib'].mean():.2f})  "
                f"Mean MAE_DBP={summary['MAE_DBP'].mean():.2f} "
                f"(zero-delta baseline={summary['MAE_DBP_baseline_calib'].mean():.2f})")

            # Pooled ME +/- SD across every held-out reading (the AAMI/ISO error
            # pair). Computed on the concatenation of all folds, not averaged
            # per-fold, so the SD reflects the true spread of the error.
            pooled = pd.concat([v["predictions"] for v in fold_results.values()],
                               axis=0, ignore_index=True)
            err_sbp = pooled["SBP_pred"] - pooled["SBP_true"]
            err_dbp = pooled["DBP_pred"] - pooled["DBP_true"]
            self.log.emit(
                f"Pooled ME +/- SD (mmHg): "
                f"SBP {err_sbp.mean():+.2f} +/- {err_sbp.std(ddof=1):.2f}   "
                f"DBP {err_dbp.mean():+.2f} +/- {err_dbp.std(ddof=1):.2f}  "
                f"(AAMI/ISO error pair; pass bar |ME|<=5 and SD<=8 mmHg)")
            self.log.emit(
                f"[context only] pop-mean baseline: Mean MAE_SBP="
                f"{summary['MAE_SBP_baseline'].mean():.2f}  Mean MAE_DBP="
                f"{summary['MAE_DBP_baseline'].mean():.2f}  "
                f"(folds beating it: {int(summary['beats_baseline'].sum())}/{len(summary)} - "
                f"not the real bar, see zero-delta above)")

            self.all_finished.emit({"summary": summary, "folds": fold_results,
                                     "bp_standards": bp_standards,
                                     "bp_standards_personalized": bp_standards_personalized})

        except Exception as e:
            tb = traceback.format_exc()
            self.error.emit(f"{e}\n\n{tb}")