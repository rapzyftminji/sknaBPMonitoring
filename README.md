# CNN-BiLSTM SKNA → BP — training app

Self-contained snapshot of the PyQt5 training application for the
CNN-BiLSTM model that maps skin sympathetic nerve activity (SKNA) + ECG to
blood pressure (SBP/DBP). Every module the app imports at runtime is
included; nothing outside this folder is needed to run it.

## Install & run

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt      # if the torch wheel fails:
                                     # pip install torch --index-url https://download.pytorch.org/whl/cpu
python main.py
```

Data/output folders (`dataset/`, `feature_result/`, `training_results/`) are
resolved from the working directory, so launch from the folder that holds them.

## Pipeline order

```
raw BIOPAC .txt export + paired vitals .csv
   ├─ recording_cutter/       (optional) trim to the real experiment window
   └─ src/upsample_txt.py     (optional) harmonise 2 kHz recordings up to 10 kHz
        └─ core_processing.py  (align, detrend, filter, window, label, features)
             ← driven by worker.py from the GUI's Signals/Windows/Features tabs
             └─ per-recording windowing_labeling/*_with_labels.csv
                  └─ aggregate_datasets.py --format npz → feature_result/combined/*.npz
                       └─ skna_dataset.py (sequence Dataset, calibration-relative targets)
                            ├─ baseline_floor.py   ← run FIRST: the MAE any model must beat
                            ├─ train_worker.py     → training_results/fold_<subj>/checkpoint.pt
                            │    └─ model.py       ← the CNN-BiLSTM itself
                            ├─ inference_worker.py ← score a checkpoint on new data
                            └─ personalize_finetune.py ← per-subject adaptation on a fold
```

## What each file is for

### Entry point

| File | Role |
| --- | --- |
| `main.py` | Launcher. Puts `src/` on `sys.path` (the modules use flat absolute imports) and calls `main_window.main()`. Always start the app with `python main.py`. |
| `requirements.txt` | numpy, pandas, scipy, matplotlib, PyQt5, pyqtgraph, PyWavelets, torch. |

### The model

| File | Role |
| --- | --- |
| `src/model.py` | **The CNN-BiLSTM.** `SKNABPModel` runs one 1D-CNN encoder per input stream (ECG, SKNA, iSKNA/rSKNA, aSKNA, plus optional FD/FFT/PSD and a 2D CNN over the CWT scalogram), concatenates the per-window embeddings into a sequence, feeds it to a **2-layer bidirectional LSTM** (hidden 64), pools across time with `AttentionPool`, and regresses `[SBP, DBP]` through a linear head. Each branch is individually switchable for ablations, and `cnn_arch` selects plain CNN / ResNet / ResNet-18 / plain ANN. Also holds the training plumbing: `TargetScaler`, `EarlyStopper`, `bp_loss`, `train_one_epoch`, `evaluate`, and `freeze_for_hybrid_calibration` (freezes everything except the last LSTM layer + attention pool + head, used by personalization). |

### Data

| File | Role |
| --- | --- |
| `src/core_processing.py` | All the signal maths, GUI-safe (no `plt.show()`, no hardcoded paths, long loops take `progress_cb`/`should_stop`): loading, Butterworth bandpass, SKNA integration, windowing, time/FFT/PSD/band features, CWT scalograms, phase statistics. |
| `src/worker.py` | `PipelineWorker(QThread)` — runs that pipeline off the UI thread, emitting a signal per stage (`load`, `preprocess`, `window`, `features`, `cwt`, `phase_stats`) and supporting mid-run cancellation. |
| `src/upsample_txt.py` | **Upstream, optional, standalone CLI.** Resamples the older 2 kHz BIOPAC exports up to the 10 kHz of the newer ones so every recording shares a rate — the pipeline takes fs from a spin-box, *not* the file header, so mixing rates silently corrupts every frequency-domain feature. Polyphase (`resample_poly`), zero-phase anti-imaging FIR; the output is a byte-compatible drop-in for `load_txt_signal`. |
| `recording_cutter/` | **Upstream, optional, separate Qt app** (`python recording_cutter/main.py`). Trims a .txt recording and/or its paired vitals .csv to a chosen time range, with a draggable region on each plot so you see the cut before saving. Self-contained — imports nothing from `src/`. |
| `src/aggregate_datasets.py` | Merges the per-recording labeled CSVs into master arrays. `--format npz` writes the `all_raw_signal.npz` / `all_skna_*.npz` / label CSVs that the training tab expects, and defines the LOSO subject groups. |
| `src/skna_dataset.py` | `SKNASequenceDataset` — the PyTorch `Dataset`. One sample = `seq_len` **consecutive** windows from a single recording (never crossing recordings/subjects); the label is the BP of the **last** window. Targets are **calibration-relative**: each recording's own first `calib_windows` windows set its baseline, and the model predicts the delta from it — the same thing a real cuffless device does. Also provides the time-split (`three_way_time_split`) and `make_loso_splits` helpers. |

### Training / evaluation

| File | Role |
| --- | --- |
| `src/train_worker.py` | `LOSOTrainingWorker(QThread)` — the training loop. Builds leave-one-subject-out folds, fits input normalization and the target scaler **on the training pool only**, trains with early stopping, and writes `training_results/fold_<subject>/checkpoint.pt` plus per-fold metrics. Emits per-epoch signals so the GUI plots loss live. |
| `src/inference_worker.py` | Loads a saved checkpoint and scores it on a held-out subject or an entirely new recording. Deliberately reuses the **checkpoint's** normalization stats and target scaler rather than refitting on the test data — refitting would leak and flatter the results. |
| `src/personalize_finetune.py` | Per-subject personalization. Splits the held-out subject's own recording by time into adapt-train `[0, 0.5)` / adapt-val `[0.5, 0.6)` / test `[0.6, 1.0]` with boundary windows dropped, scores the population model on the test slice (PRE-FT), then fine-tunes only the last LSTM layer + attention pool + head on the adapt slice and rescores (POST-FT). Runs standalone as a CLI or from the GUI's Personalize checkbox. |
| `src/baseline_floor.py` | **No-model sanity check — run this before training.** Using the identical fold definitions, it computes two floors: *pop-mean* (predict the training pool's mean BP) and *zero-delta* (predict the subject's own calibration reading forever, i.e. assume BP never changes). Zero-delta is the real bar a calibration-relative model has to clear; quote it next to any MAE. |
| `src/bp_standards.py` | Grades predicted-vs-true BP against the device-validation standards — BHS, AAMI/SP10, ISO 81060-2:2018, IEEE 1708. Pure scoring, trains nothing. |

### GUI

| File | Role |
| --- | --- |
| `src/main_window.py` | The whole PyQt5 window. Tabs: **Signals**, **Windows**, **Features**, **Phase Comparison**, **Model Training** (fold config, hyperparameters, personalization knobs, live loss curves), **Test Saved Model** (point at a checkpoint + dataset, get metrics and standards grades). It owns the widgets and threads only — no maths lives here. |

---

# Function reference

Every module, function by function. The table above says *what each file is
for*; this section says *what is inside it*.

## 1. Entry point

**`main.py`** — inserts `src/` on `sys.path` (the modules use flat absolute
imports), pre-imports torch defensively, calls `main_window.main()`.

## 2. Preprocessing — `src/core_processing.py`

The maths library. GUI-safe throughout: `matplotlib.use("Agg")`, no
`plt.show()`, no hardcoded paths, long loops take `progress_cb` / `should_stop`.

**Load & align**

- `load_txt_signal` / `load_csv_vitals` — BIOPAC .txt (skiprows=14, columns
  Time/CH1/CH40/CH41) and the paired vitals .csv.
- `_ecg_to_hr`, `_hr_xcorr_lag`, `_vitals_rel_seconds`, `_vitals_time_and_hr` —
  derive HR from both sources so they can be matched on a common quantity.
- `estimate_bp_alignment_offset` — cross-correlates the two HR traces (±300 s)
  and returns the lag plus its correlation.
- `align_time_axes` — applies that lag (`method="hr_xcorr"`). This is the fix
  for the old fixed-offset assumption, and it runs on every pass.

**Condition**

- `normalize_minmax` / `normalize_log_minmax` — the log variant compresses
  before min–max; it is the correct one for FFT/PSD magnitudes.
- `detrend_full_signal`, `detrend_channels`, `detrend_signal`.
- `highpass_filter`, `highpass_channels` — Butterworth.
- `preprocess_skna` — bandpass → rectify (`rect`) → integrate, producing the
  SKNA / iSKNA / rSKNA / aSKNA streams.

**Window & label**

- `window_raw_signal` — slice the conditioned signal into fixed windows.
- `_find_vital_col`, `label_windows_with_bp` — attach SBP/DBP to each window.
- `build_window_labels`, `write_labels_csv`.
- `_build_labeled_csv`, `_build_labeled_npz`, `save_labeled_datasets`,
  `save_labeled_datasets_npz`, `load_labeled_npz` — write
  `windowing_labeling/*_with_labels.csv` and the matching `.npz` arrays.

**Features**

- `extract_all_time_features` — time-domain statistics per window.
- `extract_fft_features`, `extract_psd_features` (Welch).
- `extract_fftband_features`, `extract_psdband_features` — per-band power shares.
- `extract_top_frequency_features`.
- `extract_cwt_features`, `compute_single_cwt`, `compute_cwt_array`,
  `extract_cwt_arrays`, `_pool_columns_mean` — wavelet scalograms, both scalar
  features and the full 2D arrays for the CNN2D branch.
- `_fd_band_slice`, `compute_fd_array`, `extract_fd_arrays` — the
  frequency-domain arrays consumed by the FD / FFT / PSD branches.
- `_phase_labels`, `extract_phase_delta_features`, `extract_recovery_features` —
  cold-pressor phase contrasts.
- `calculate_phase_stats`, `fig_statistical_comparison`,
  `fig_continuous_segmented_signals` — statistics and the two comparison figures.

### `src/worker.py`

`PipelineWorker(QThread)`. `run()` drives everything above off the UI thread,
emitting `stage_started` / `stage_finished` for
`load → detrend → preprocess → window → label → features → cwt → cwt_arrays →
fd_arrays → phase_stats`. `request_stop()` / `_should_stop()` /
`_finish_cancelled()` give clean mid-run cancellation.

### `src/upsample_txt.py` (standalone CLI)

`read_header`, `parse_interval_ms`, `patch_header`, `upsample_file` (polyphase
`resample_poly`, zero-phase anti-imaging FIR, `'line'` edge padding so the BP
channel's large DC offset does not ring), `default_out_path`, `main`. The output
is a byte-compatible drop-in for `load_txt_signal`.

### `recording_cutter/`

`main.py` plus `cutter_window.py` (`RecordingCutterWindow`): trim a .txt and/or
its paired .csv with a draggable region on each plot. It never trusts the raw
time column's unit (inconsistent across recording batches) — it parses `fs` from
the header and uses `sample_index / fs`.

## 3. Aggregation — `src/aggregate_datasets.py`

- `_find_recording_dirs` — discover the `<subject>/windowing_labeling/` folders.
- `aggregate_one_kind` / `aggregate_all` — master CSVs.
- `aggregate_one_kind_npz` / `aggregate_all_npz` — `all_raw_signal.npz`,
  `all_skna_signal.npz`, `all_iskna_rskna.npz`, `all_askna.npz` and their label
  CSVs. This is what `--format npz` runs, and what the training tab expects.
- `load_npz_dataset` — reload a labels-CSV + npz pair.
- `get_loso_groups` — the leave-one-subject-out group vector.
- `summarize_subjects`, `flag_short_recordings` — pre-training sanity checks.

## 4. Dataset — `src/skna_dataset.py`

### `SKNASequenceDataset`

One sample = `seq_len` consecutive windows from a single recording, labeled with
the BP of the **last** window.

- `__init__` — loads all four npz streams plus labels; optional decimation.
- `_decimate_time_axis` — downsample the within-window axis.
- `_compute_baselines` — per-`(Subject_ID, Session_ID)` baseline SBP/DBP from
  that recording's own first `calib_windows` windows.
- `_compute_calib_norm_stats` — per-recording input normalization from that same
  calibration slice.
- `_build_sequences` — builds sequences that **never cross a recording
  boundary**, so no sample mixes two subjects or sessions.
- `_fit_norm_stats` / `subset(seq_indices, fit_norm, norm_stats)` — fit on the
  training pool only; subsets inherit the passed-in stats.
- `split_within_subjects` — train/val split inside the training pool.
- `sequence_groups` — the per-sequence subject id, for grouping.
- `__getitem__` — returns `ecg / skna / rskna / iskna / askna` as
  `(seq_len, 1, window_size)` tensors plus `sbp`, `dbp`, `baseline_sbp`,
  `baseline_dbp`. Under `norm_mode="calib"` it reads that recording's own stats
  off the target window; otherwise it uses the global training stats.

### Module functions

- `make_loso_splits` — leave-one-subject-out fold indices.
- `three_way_time_split(dataset, subject, adapt_end=0.5, val_end=0.6)` —
  adapt / val / test by time. A sequence counts for a bucket only if **all** of
  its windows fall inside it, so straddling sequences are dropped and there is
  zero sliding-window overlap between the slices. Falls back to carving val from
  the tail of adapt when a short recording leaves the middle band empty.

## 5. The model — `src/model.py`

### Branches

Each encodes one window. All subclass `_CheckpointedBranch`, which
gradient-checkpoints above `CNN_CHECKPOINT_ABOVE = 256` timesteps; `_bn_safe`
keeps BatchNorm running stats correct under that checkpointing.

- `CNNBranch` — plain 1D CNN, conv → BN → ReLU → pool.
- `CNN2DBranch` — plain 2D CNN for the CWT scalogram.
- `ANNBranch` — adaptive-avg-pool to `pool_len`, then an MLP (the no-convolution
  control).
- `_BasicBlock1D` / `ResNet1DBranch` and `_BasicBlock2D` / `CNN2DResNetBranch` —
  the residual variants; `_expand_blocks` sets depth per stage (ResNet-18 = 2).

Every branch is sized so `out_dim` matches regardless of architecture, so the
fused LSTM input width never changes between variants.

### `AttentionPool`

`Linear → Tanh → Linear` score per timestep, softmax **over time**, weighted sum
to `(B, D)`. Can return the weights, which show which windows the model leaned on.

### `SKNABPModel` — the CNN-BiLSTM

1. `_apply_branch` reshapes `(B,T,C,L) → (B·T,C,L)`, encodes every window, and
   reshapes back (`_apply_branch_2d` does the same for the scalogram).
2. Concatenates all enabled branch outputs into `feat_dim`.
3. `nn.LSTM(feat_dim, hidden_size=64, num_layers=2, bidirectional=True,
   batch_first=True)`.
4. `AttentionPool(128)` → dropout → `Linear(128, 2)` = **[Δ SBP, Δ DBP]**.

Branches are individually switchable (`use_ecg`, `use_skna`, `use_iskna_rskna`,
`use_askna`, `use_fd` / `use_fft` / `use_psd` / `use_cwt`); a disabled branch is
not built at all. `cnn_arch ∈ {plain, resnet, resnet18, ann}` picks the 1D
architecture and `cwt_arch` picks the 2D half independently. Disabling every
branch raises.

### Training plumbing

- `TargetScaler` — `fit` / `transform` / `inverse_transform` / `state_dict` /
  `from_state_dict`; saved into every checkpoint.
- `batch_to_device`.
- `run_batch` — computes `target_delta = target_abs − baseline`, scales it, and
  returns `(pred, target_delta_norm, target_delta, target_abs, baseline)`. **The
  network predicts the delta from calibration, not absolute mmHg**; absolute BP
  is recovered by adding the baseline back.
- `bp_loss` — smooth-L1, plus an optional anti-collapse term
  `λ · relu(std(target) − std(pred))` that penalises the predict-the-mean
  failure mode.
- `train_one_epoch`, `evaluate(..., return_predictions=…)` — loss plus MAE in
  both delta and absolute mmHg.
- `EarlyStopper(patience, min_delta).step(value)`.
- `freeze_for_hybrid_calibration(model)` — freezes everything except the **last
  LSTM layer (both directions), the attention pool and the head**;
  `unfreeze_all` reverses it.

## 6. Training & evaluation

### `src/train_worker.py`

`LOSOTrainingWorker(QThread)`, one large `run()`: reads the parameter dict
(architecture, branch toggles, hyperparameters, personalization knobs), builds
the dataset, then per LOSO fold fits norm stats and the `TargetScaler` **on the
training pool only**, trains with early stopping, evaluates the held-out
subject, optionally runs the personalization fine-tune, and writes
`training_results/fold_<subject>/checkpoint.pt`. After all folds it pools every
held-out reading and grades the pool — pooled ME ± SD is computed on the
concatenation, not averaged per fold, so the SD reflects the true error spread.
Signals: `log`, `fold_started`, `epoch_progress`, `fold_finished`,
`all_finished`, `error`.

### `src/inference_worker.py`

`InferenceWorker(QThread)`. Loads a checkpoint and scores it on a held-out
subject or an entirely new recording, reusing the **checkpoint's** normalization
stats and target scaler. Refitting on the test set would leak and flatter the
numbers.

### `src/personalize_finetune.py`

Standalone CLI mirroring the GUI's Personalize path.

- `build_dataset`, `build_model` — reconstruct the dataset and architecture from
  the checkpoint's saved params.
- `run(args)` — per subject: three-way time split → evaluate the base model on
  the test slice (**PRE-FT**) → `freeze_for_hybrid_calibration` → fine-tune on
  the adapt slice with early stopping on val → re-evaluate (**POST-FT**).
- `zero_delta_baseline(test_metrics)` — the same bar computed on that same
  slice, so PRE/POST are never read without it.

### `src/baseline_floor.py`

`compute_floor(combined_dir, seq_len, stride, calib_windows, val_frac)` — the
identical fold construction with no network at all. Reports **pop-mean**
(predict the training pool's mean BP) and **zero-delta** (predict the subject's
own calibration reading forever). Zero-delta is the real bar a
calibration-relative model must clear; run this before spending time training.

### `src/bp_standards.py`

Grading only — nothing here fits or trains.

- `bhs_grade` — A/B/C/D from the ≤5 / ≤10 / ≤15 mmHg cumulative percentages.
- `aami_sp10` — mean error ± SD against the SP10 limits.
- `iso_81060_2` with `_iso_c2_sd_limit` — criterion 1 and the subject-wise
  criterion 2.
- `ieee_grade` — IEEE 1708 A–D.
- `compute_bp_standards(pred, true, subject_ids)`, `compute_bp_standards_from_df`.
- `format_standards_report(standards, zero_delta_mae=None)` — prints the grade
  next to the baseline it has to clear.
- `_pool_predictions`, `standards_summary_rows`, `main` — pool the fold
  checkpoints and grade from the command line.

## 7. GUI — `src/main_window.py`

- `PandasModel(QAbstractTableModel)` — DataFrames into a `QTableView`.
- `downsample_for_plot(x, y, max_points=20000)` — keeps long recordings
  responsive to pan/zoom.
- `MplCanvas(FigureCanvas)` — embedded matplotlib canvas.
- `MainWindow(QMainWindow)` — the whole application. Tabs: **Signals**,
  **Windows**, **Features**, **Phase Comparison**, **Model Training** (fold
  config, hyperparameters, branch toggles, personalization knobs, live loss
  curves), **Test Saved Model** (checkpoint + dataset → metrics and standards
  grades). Owns widgets and threads only; no maths lives here.
- `main()` — builds the `QApplication` and shows the window.

## Two things that are easy to misread

1. **The network outputs a delta from each recording's own calibration
   baseline, not absolute mmHg.** Any MAE therefore has to be quoted against
   `baseline_floor.py`'s zero-delta number, which is what "predict no change
   since calibration" already achieves.
2. **The personalized metrics live on a smaller held-out slice** (the last 40%
   of each subject's recording), so they are comparable to the `p_base_*`
   columns on that same slice — never to the plain whole-subject LOSO columns.
