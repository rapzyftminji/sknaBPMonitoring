"""
Load a trained checkpoint (saved per-fold by train_worker.py) and evaluate it
on a dataset - either a subject held out during training, or an entirely new
recording collected later. No training happens here, this is inference/testing
only.

Critically, this reuses the CHECKPOINT's input normalization stats and target
scaler (not stats fit on the test data) - evaluating with stats fit on the
test set itself would silently leak information and make results look better
than they'd be in real deployment.
"""

import os
import traceback

import numpy as np
import pandas as pd
from PyQt5.QtCore import QThread, pyqtSignal


class InferenceWorker(QThread):
    log = pyqtSignal(str)
    finished_ok = pyqtSignal(object)   # {"metrics": {...}, "predictions": df}
    error = pyqtSignal(str)

    def __init__(self, params, parent=None):
        super().__init__(parent)
        self.params = params

    def run(self):
        try:
            try:
                import torch
                from torch.utils.data import DataLoader
                from skna_dataset import SKNASequenceDataset
                from model import SKNABPModel, TargetScaler, evaluate, ANN_POOL_LEN
            except (OSError, ImportError) as e:
                self.error.emit(
                    "Failed to load PyTorch (environment/install issue, not a bug in "
                    "this pipeline's code):\n\n" + str(e) +
                    "\n\nSee the Model Training tab's error message for troubleshooting steps."
                )
                return

            p = self.params
            checkpoint_path = p["checkpoint_path"]
            combined_dir = p["combined_dir"]
            test_subjects = p.get("test_subjects") or None   # None/[] = use every subject found
            allow_in_sample = p.get("allow_in_sample", False)
            batch_size = p.get("batch_size", 16)
            device = "cuda" if (p.get("use_gpu", True) and torch.cuda.is_available()) else "cpu"

            self.log.emit(f"Device: {device}")
            self.log.emit(f"Loading checkpoint: {checkpoint_path}")
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
            train_params = ckpt.get("params", {})

            seq_len = train_params.get("seq_len", 10)
            # Stride can be overridden here (e.g. stride=seq_len for non-overlapping
            # windows at test time), independent of what was used during training.
            stride = p.get("stride") or train_params.get("stride", 1)
            # Must match training: the model was trained on signals decimated by
            # this factor, so inference has to decimate identically or the CNN sees
            # a different sample rate / window length than it was fit on.
            downsample = int(train_params.get("downsample", 1) or 1)

            self.log.emit(f"Loading dataset from {combined_dir} "
                           f"(seq_len={seq_len} from checkpoint, stride={stride}"
                           + (f", downsample={downsample}x from checkpoint" if downsample > 1 else "")
                           + ")...")
            self.log.emit("Reusing the CHECKPOINT's normalization stats (not re-fitting on "
                           "this data) so results reflect real deployment, not leakage.")
            # Feed the optional FD/CWT branches only if this checkpoint was
            # trained with them (recorded in its params). A checkpoint that used
            # them but whose aggregated file is missing here is a clear error.
            use_fd = train_params.get("use_fd", False)
            use_fft = train_params.get("use_fft", False)
            use_psd = train_params.get("use_psd", False)
            use_cwt = train_params.get("use_cwt", False)
            fd_npz = cwt_npz = None
            if use_fd or use_fft or use_psd:
                fd_npz = os.path.join(combined_dir, "all_fd_signal.npz")
                if not os.path.isfile(fd_npz):
                    raise FileNotFoundError(
                        f"This checkpoint used an FD/FFT/PSD branch but {fd_npz} is missing. "
                        "Compute FD arrays for this dataset and re-aggregate before evaluating.")
            if use_cwt:
                cwt_npz = os.path.join(combined_dir, "all_cwt_signal.npz")
                if not os.path.isfile(cwt_npz):
                    raise FileNotFoundError(
                        f"This checkpoint used the CWT branch but {cwt_npz} is missing. "
                        "Compute CWT arrays for this dataset and re-aggregate before evaluating.")

            ds = SKNASequenceDataset(
                raw_signal_npz=os.path.join(combined_dir, "all_raw_signal.npz"),
                raw_signal_labels_csv=os.path.join(combined_dir, "all_raw_signal_labels.csv"),
                skna_signal_npz=os.path.join(combined_dir, "all_skna_signal.npz"),
                iskna_rskna_npz=os.path.join(combined_dir, "all_iskna_rskna.npz"),
                askna_npz=os.path.join(combined_dir, "all_askna.npz"),
                fd_signal_npz=fd_npz, cwt_signal_npz=cwt_npz,
                seq_len=seq_len, stride=stride, downsample=downsample,
                norm_stats=ckpt["norm_stats"],
            )
            groups = ds.sequence_groups()
            available_subjects = sorted(set(groups))
            self.log.emit(f"Dataset has {len(ds)} sequences across subjects: {available_subjects}")

            requested = list(test_subjects) if test_subjects else list(available_subjects)
            if test_subjects:
                unknown = [s for s in requested if s not in available_subjects]
                if unknown:
                    self.log.emit(f"[warning] requested subject(s) not found in this dataset "
                                   f"and will be skipped: {unknown}")
                    requested = [s for s in requested if s not in unknown]

            # Leakage guard: this checkpoint's own training fold saw every subject
            # in ckpt["train_subjects"] except ckpt["test_subject"]. Evaluating it
            # on a subject it already trained on is in-sample recall, not a real
            # test - that's what made an earlier "all subjects, blank filter" run
            # look far better than the true LOSO numbers.
            held_out_subject = ckpt.get("test_subject")
            train_subjects = set(ckpt.get("train_subjects") or [])
            if train_subjects:
                in_sample = [s for s in requested if s in train_subjects]
                if in_sample and not allow_in_sample:
                    requested = [s for s in requested if s not in in_sample]
                    self.log.emit(
                        f"[leakage guard] this checkpoint was trained on {sorted(train_subjects)} "
                        f"(held out {held_out_subject!r}). Dropping in-sample subject(s) "
                        f"{in_sample} from evaluation - the model already saw their data during "
                        "training, so metrics on them would be memorization, not a real test. "
                        "Pass allow_in_sample=True to force-include them anyway (debugging only)."
                    )
                elif in_sample and allow_in_sample:
                    self.log.emit(
                        f"[warning] allow_in_sample=True: including in-sample subject(s) "
                        f"{in_sample} anyway. Their metrics reflect memorization, not "
                        "generalization - do not report them as test performance."
                    )
            else:
                self.log.emit(
                    "[leakage guard] this checkpoint has no recorded train_subjects/test_subject "
                    "(older checkpoint, saved before this guard existed) - cannot verify whether "
                    "the requested subjects were seen during training. Only trust these numbers "
                    "if you already know which subject this fold held out."
                )

            if not requested:
                raise ValueError(
                    "No subjects left to evaluate after dropping in-sample subject(s). "
                    f"This checkpoint's held-out subject is {held_out_subject!r} - test on "
                    "that, or point at a brand-new recording never used in training."
                )

            mask = np.isin(groups, requested)
            if not mask.any():
                raise ValueError(f"None of the requested subjects {requested} are in "
                                  f"this dataset. Available: {available_subjects}")
            eval_ds = ds.subset(np.where(mask)[0], norm_stats=ds.norm_stats)
            self.log.emit(f"Evaluating on subject(s): {sorted(set(np.asarray(groups)[mask]))}")

            # Fallbacks are the OLD 2-CNN / 1-BiLSTM architecture on purpose:
            # checkpoints saved before the deeper-model change don't record these
            # keys, so they must reconstruct with their original shapes or
            # load_state_dict fails. New checkpoints save ecg_channels/lstm_layers
            # explicitly (see train_worker), so they rebuild at the right depth.
            model = SKNABPModel(
                ecg_channels=tuple(train_params.get("ecg_channels", (16, 32))),
                skna_channels=tuple(train_params.get("skna_channels", (16, 32))),
                iskna_rskna_channels=tuple(train_params.get("iskna_rskna_channels", (16, 32))),
                lstm_hidden=train_params.get("lstm_hidden", 64),
                lstm_layers=train_params.get("lstm_layers", 1),
                dropout=train_params.get("dropout", 0.3),
                # Default on so pre-ablation checkpoints reconstruct unchanged.
                use_ecg=train_params.get("use_ecg", True),
                use_skna=train_params.get("use_skna", True),
                use_iskna_rskna=train_params.get("use_iskna_rskna", True),
                use_askna=train_params.get("use_askna", True),
                # Default off so pre-FD/CWT checkpoints reconstruct unchanged.
                use_fd=use_fd, fd_channels=tuple(train_params.get("fd_channels", (16, 32, 64))),
                use_fft=use_fft, fft_channels=tuple(train_params.get("fft_channels", (16, 32, 64))),
                use_psd=use_psd, psd_channels=tuple(train_params.get("psd_channels", (16, 32, 64))),
                use_cwt=use_cwt, cwt_channels=tuple(train_params.get("cwt_channels", (8, 16, 32))),
                # Must match training or the module names/shapes won't line up in
                # load_state_dict; default "plain" so pre-ResNet checkpoints load unchanged.
                cnn_arch=train_params.get("cnn_arch", "plain"),
                # 2D branch arch; absent (pre-cwt_arch checkpoints) => follow cnn_arch,
                # which is exactly what those checkpoints were built with.
                cwt_arch=train_params.get("cwt_arch"),
                # Depth of the "resnet" branches; "resnet18" ignores it (fixed at 2).
                # Default 1 so pre-existing resnet checkpoints reconstruct unchanged.
                resnet_blocks_per_stage=train_params.get("resnet_blocks_per_stage", 1),
                # Pool length of the "ann" branches; ignored by the conv archs.
                ann_pool_len=train_params.get("ann_pool_len", ANN_POOL_LEN),
            ).to(device)
            model.load_state_dict(ckpt["model_state"])
            target_scaler = TargetScaler.from_state_dict(ckpt["target_scaler"])

            loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False)
            self.log.emit(f"Running inference on {len(eval_ds)} sequences...")
            metrics = evaluate(model, loader, device, target_scaler, return_predictions=True,
                                anti_collapse_lambda=train_params.get("anti_collapse_lambda", 0.0))

            pred_df = pd.DataFrame({
                "Subject_ID": metrics["subject_id"],
                "SBP_true": metrics["true"][:, 0], "SBP_pred": metrics["pred"][:, 0],
                "DBP_true": metrics["true"][:, 1], "DBP_pred": metrics["pred"][:, 1],
            })

            self.log.emit(
                f"Result: MAE_SBP={metrics['MAE_SBP']:.2f}  MAE_DBP={metrics['MAE_DBP']:.2f}  "
                f"corr_SBP={metrics['corr_SBP']:.3f}  corr_DBP={metrics['corr_DBP']:.3f}"
            )

            self.finished_ok.emit({"metrics": metrics, "predictions": pred_df})

        except Exception as e:
            tb = traceback.format_exc()
            self.error.emit(f"{e}\n\n{tb}")