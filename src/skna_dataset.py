"""
PyTorch Dataset for the SKNA/ECG -> SBP/DBP model.

Your architecture runs a BiLSTM over a SEQUENCE of consecutive 5s windows
(T timesteps) and outputs a single SBP/DBP prediction for that sequence -
it does not predict one BP value per individual window. So each training
sample here is a short run of `seq_len` consecutive windows taken from a
single recording (never spanning two different recordings/subjects), with
the target being the SBP/DBP of the LAST window in that sequence (i.e.
"given the last `seq_len` windows of history, predict BP right now").

Targets are CALIBRATION-RELATIVE, not raw absolute mmHg: each recording
(Subject_ID, Session_ID) gets its own baseline SBP/DBP from the mean of its
own first `calib_windows` windows - exactly like a real cuffless-BP device
being calibrated once against a cuff reading before use. The model is
trained/evaluated on delta = absolute BP - that recording's own baseline
(see model.py's run_batch/evaluate). This baseline uses ONLY that same
recording's own early samples, so computing it for a held-out LOSO test
subject is not leakage - it mirrors what a real device would have (its own
calibration reading), not information borrowed from other subjects. This
matters because absolute BP level is driven heavily by inter-subject factors
(arterial stiffness, age, etc.) that a few minutes of SKNA/ECG can't infer
for someone the model has never seen - see the mean-collapse diagnosis in
train_worker.py's trivial-baseline logging. Sequences whose target window
falls inside the calibration segment itself are excluded (predicting your
own calibration reading back is trivial and would inflate metrics).

Usage:
    from skna_dataset import SKNASequenceDataset, make_loso_splits

    ds_full = SKNASequenceDataset(
        raw_signal_npz="all_raw_signal.npz",
        raw_signal_labels_csv="all_raw_signal_labels.csv",
        skna_signal_npz="all_skna_signal.npz",
        iskna_rskna_npz="all_iskna_rskna.npz",
        askna_npz="all_askna.npz",
        seq_len=10, stride=1,
    )

    for train_idx, test_idx, held_out_subject in make_loso_splits(ds_full):
        train_ds = ds_full.subset(train_idx, fit_norm=True)   # stats computed from train only
        test_ds = ds_full.subset(test_idx, norm_stats=train_ds.norm_stats)
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
        ...
"""

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SKNASequenceDataset(Dataset):
    def __init__(self, raw_signal_npz, raw_signal_labels_csv, skna_signal_npz,
                 iskna_rskna_npz, askna_npz, seq_len=10, stride=1,
                 group_cols=("Subject_ID", "Session_ID"), calib_windows=10,
                 fd_signal_npz=None, cwt_signal_npz=None, norm_mode="global",
                 downsample=1,
                 _preloaded=None, _seq_index=None, norm_stats=None):
        """
        seq_len: number of consecutive windows per training sample (each window
                 is 3s, so seq_len=10 -> 30s of context per prediction).
        stride: step between consecutive sequence start points. stride=1 gives
                maximum overlap/data reuse; larger stride gives more independent
                (less correlated) sequences at the cost of fewer samples.
        group_cols: columns that uniquely identify one continuous recording;
                    sequences never cross a group boundary.
        calib_windows: number of each recording's own first windows averaged
                       into its calibration baseline SBP/DBP (10 windows = 30s
                       at the default 3s/window). Sequences whose target falls
                       within this calibration segment are excluded.
        norm_mode: how input signals are z-scored before hitting the CNN branches.
                   "global" - one mean/std per channel, fit over the whole TRAIN
                       subset and reused for val/test (the original behaviour).
                       Every subject's windows are forced through the training
                       population's scale.
                   "calib" - each recording is z-scored by the mean/std of its
                       OWN first `calib_windows` windows, exactly mirroring how
                       `_compute_baselines` derives that recording's BP baseline.
                       This is the input-side counterpart of the calibration-
                       relative target: a held-out subject whose SKNA/ECG sits at
                       a different absolute scale is no longer misread through
                       another population's statistics. Uses only that
                       recording's own early samples, so it is not LOSO leakage
                       for the same reason the BP baseline isn't.
        downsample: integer factor by which the 1D TIME-DOMAIN signals
                    (ECG/SKNA/iSKNA/rSKNA) are decimated along their sample axis
                    at load time. 1 = off (full resolution). Applied ONCE here,
                    in memory, so callers never have to pre-downsample each
                    recording's file. An anti-aliased zero-phase FIR low-pass is
                    applied before subsampling, so no aliasing - it cleanly
                    discards signal above the new Nyquist ((fs/2)/factor). At
                    fs=10 kHz, factor=5 -> 2 kHz (keeps 0-1000 Hz, the classic
                    sKNA band); factor=10 -> 1 kHz (drops the 500-1000 Hz sKNA
                    band, so only sensible for ECG-only runs). Cuts both resident
                    RAM and CNN activation memory by ~`factor`. Per-window scalars
                    (aSKNA) and spectral features (FD/CWT) are NOT decimated -
                    they have no raw time axis. The factor is recorded in the
                    training params so inference decimates identically.
        """
        if norm_mode not in ("global", "calib"):
            raise ValueError(f"norm_mode must be 'global' or 'calib', got {norm_mode!r}.")
        if int(downsample) < 1:
            raise ValueError(f"downsample must be a positive integer factor, got {downsample!r}.")
        downsample = int(downsample)
        if _preloaded is not None:
            # Internal fast path used by .subset() - reuse already-loaded arrays.
            self._arrays = _preloaded
        else:
            # Memory-lean load. Each core signal is an ~(n_windows x window_len)
            # float32 array - for 5 s / 10 kHz windows that's ~1 GB EACH, so the
            # four core signals are ~4 GB resident. The previous version loaded
            # every raw array AND simultaneously built a reordered copy of each
            # (ecg[order], skna[order], ...), so at peak both the originals and
            # the copies were alive at once - ~8 GB - which OOM-killed the process
            # on large datasets (and, when launched from inside VS Code, took the
            # editor down with it via the shared cgroup). Here we compute the sort
            # order first, then load -> reorder -> free ONE array at a time, so
            # only a single raw array's transient copy is ever alive on top of the
            # already-finished ones. Result is byte-identical to before.
            labels = pd.read_csv(raw_signal_labels_csv)
            n = len(labels)

            # Sort by (group_cols, Window_ID) so each recording's windows are
            # contiguous and in time order - required for valid sequences.
            labels = labels.reset_index(drop=True)
            sort_cols = [c for c in group_cols if c in labels.columns] + ["Window_ID"]
            order = labels.sort_values(sort_cols, na_position="first").index.to_numpy()
            sorted_labels = labels.iloc[order].reset_index(drop=True)

            def _load_sorted(path, key, decimate=False):
                """Load one npz array, return it already reordered by `order`, and
                free the raw (pre-reorder) array before the next one is loaded, so
                two full ~1 GB signal arrays never coexist in memory.

                decimate=True downsamples the time axis by `downsample` FIRST (on
                the full raw array, which is then freed), so the reorder and the
                stored copy are already the smaller ~1/factor size - this also
                keeps the transient peak to one raw array at a time. Decimation is
                per-row along the sample axis, so it commutes with the row reorder;
                doing it before `[order]` is identical to doing it after."""
                with np.load(path) as npz:
                    arr = npz[key]
                    if arr.shape[0] != n:
                        raise ValueError(
                            f"Row count mismatch: labels has {n} rows but '{key}' in "
                            f"{os.path.basename(path)} has {arr.shape[0]}. The arrays were "
                            f"likely computed for only some recordings - recompute for every "
                            f"recording (or leave that branch off) so all inputs stay aligned "
                            f"window-for-window.")
                    if decimate and downsample > 1:
                        arr = self._decimate_time_axis(arr, downsample)
                    return arr[order]

            # Only the 1D time-domain waveforms are decimated (they carry the raw
            # sample axis that the CNN convolves over and that dominates memory).
            ecg = _load_sorted(raw_signal_npz, "X", decimate=True)
            skna = _load_sorted(skna_signal_npz, "X", decimate=True)
            # rSKNA and iSKNA share one .npz; load them in separate passes so both
            # raw ~1 GB arrays are never decompressed into memory at the same time.
            rskna = _load_sorted(iskna_rskna_npz, "rSKNA", decimate=True)
            iskna = _load_sorted(iskna_rskna_npz, "iSKNA", decimate=True)
            # aSKNA is one scalar per window (no time axis) - never decimated.
            askna = _load_sorted(askna_npz, "X")

            # Optional extra branches (frequency-domain FFT+PSD, and CWT
            # scalogram). Only loaded when a path is given; each must have the
            # same per-window row count and ordering as the core signals, or a
            # sequence would silently pull mismatched windows. CWT is stored as
            # uint8 (0-255) on disk to save space; reordered as uint8 (cheap) then
            # rescaled to float [0, 1] so everything downstream is uniform float.
            fd = _load_sorted(fd_signal_npz, "X") if fd_signal_npz is not None else None
            cwt = None
            if cwt_signal_npz is not None:
                cwt = _load_sorted(cwt_signal_npz, "X").astype(np.float32) / 255.0

            group_keys_bl = [c for c in group_cols if c in sorted_labels.columns]
            baseline_sbp, baseline_dbp = self._compute_baselines(sorted_labels, group_keys_bl, calib_windows)

            # Arrays are already in sorted order (reordered inside _load_sorted),
            # so they are stored as-is - no second reordered copy is built here.
            self._arrays = dict(
                labels=sorted_labels,
                ecg=ecg, skna=skna, rskna=rskna,
                iskna=iskna, askna=askna,
                baseline_sbp=baseline_sbp, baseline_dbp=baseline_dbp,
            )
            if fd is not None:
                self._arrays["fd"] = fd
            if cwt is not None:
                self._arrays["cwt"] = cwt

            # Per-recording calibration stats live alongside the arrays so every
            # .subset() inherits them - they describe the RECORDING, not the split.
            self._arrays["calib_norm"] = self._compute_calib_norm_stats(
                self._arrays, group_keys_bl, calib_windows)

        labels = self._arrays["labels"]

        if _seq_index is not None:
            self.seq_index = _seq_index
        else:
            group_keys = [c for c in group_cols if c in labels.columns]
            self.seq_index = self._build_sequences(labels, group_keys, seq_len, stride, calib_windows)

        self.seq_len = seq_len
        self.calib_windows = calib_windows
        self.norm_mode = norm_mode
        self.downsample = downsample
        self.norm_stats = norm_stats
        if norm_stats is None:
            self._fit_norm_stats()

    @staticmethod
    def _decimate_time_axis(arr, factor):
        """Anti-aliased downsample along the last (sample/time) axis by an integer
        `factor`. A zero-phase FIR low-pass removes everything above the new
        Nyquist BEFORE subsampling, so there is no aliasing - the decimated signal
        simply no longer contains content above (fs/2)/factor. Returns a
        contiguous float32 array (torch.from_numpy needs contiguous; float32 keeps
        memory at 1/factor rather than letting scipy widen it to float64)."""
        if factor <= 1:
            return arr
        from scipy.signal import decimate
        out = decimate(arr, factor, axis=-1, ftype="fir", zero_phase=True)
        return np.ascontiguousarray(out, dtype=np.float32)

    @staticmethod
    def _compute_baselines(labels, group_keys, calib_windows):
        """
        Per-recording calibration baseline: mean SBP/DBP of the first
        `calib_windows` time-ordered rows of each (Subject_ID, Session_ID)
        group. Only ever uses that SAME recording's own early samples - like
        a real cuffless-BP device's one-time cuff calibration - so it carries
        zero cross-subject information and is safe to compute even for a
        held-out LOSO test subject. Returns (baseline_sbp, baseline_dbp),
        each an array of length len(labels) broadcasting one value per row.
        """
        n = len(labels)
        baseline_sbp = np.empty(n, dtype=np.float64)
        baseline_dbp = np.empty(n, dtype=np.float64)
        fillna_labels = labels.copy()
        for c in group_keys:
            fillna_labels[c] = fillna_labels[c].fillna("")
        for _, group_df in fillna_labels.groupby(group_keys, sort=False):
            positions = group_df.index.to_numpy()   # time-sorted & contiguous
            calib_pos = positions[:calib_windows]
            baseline_sbp[positions] = labels["SBP"].iloc[calib_pos].mean()
            baseline_dbp[positions] = labels["DBP"].iloc[calib_pos].mean()
        return baseline_sbp, baseline_dbp

    # Channels that get z-scored on the way into the model. Kept in one place so
    # the global and per-calibration paths can never drift out of sync.
    NORM_CHANNELS = ("ecg", "skna", "rskna", "iskna", "askna", "fd", "cwt")

    @classmethod
    def _compute_calib_norm_stats(cls, arrays, group_keys, calib_windows):
        """
        Per-recording input-normalization stats: for each signal channel, the
        mean/std over that (Subject_ID, Session_ID) recording's own first
        `calib_windows` windows. Returns {channel: (mean_col, std_col)} where
        each column is length len(labels), broadcasting one recording's value to
        all of its rows (same layout as _compute_baselines' output).

        Like the BP baseline, this only ever reads a recording's own early
        samples, so it carries zero cross-subject information and is safe to
        compute for a held-out LOSO test subject - it is exactly what a real
        device would have after its one-time calibration.

        A recording whose calibration segment is (near-)constant in some channel
        would give std ~ 0 and blow up the division, so those fall back to that
        channel's global std over the whole dataset.
        """
        labels = arrays["labels"]
        n = len(labels)
        fillna_labels = labels.copy()
        for c in group_keys:
            fillna_labels[c] = fillna_labels[c].fillna("")

        stats = {}
        for key in cls.NORM_CHANNELS:
            if key not in arrays:
                continue                      # optional fd/cwt branch not loaded
            arr = arrays[key]
            global_std = float(arr.std())
            if global_std < 1e-8:
                global_std = 1.0
            mean_col = np.empty(n, dtype=np.float64)
            std_col = np.empty(n, dtype=np.float64)
            for _, group_df in fillna_labels.groupby(group_keys, sort=False):
                positions = group_df.index.to_numpy()   # time-sorted & contiguous
                calib_pos = positions[:calib_windows]
                m = float(arr[calib_pos].mean())
                s = float(arr[calib_pos].std())
                mean_col[positions] = m
                std_col[positions] = s if s > 1e-8 else global_std
            stats[key] = (mean_col, std_col)
        return stats

    @staticmethod
    def _build_sequences(labels, group_keys, seq_len, stride, calib_windows=0):
        """Return an (num_sequences, seq_len) int array of row-positions, one row per sequence.
        Sequences whose target (last) position falls within the recording's own
        first `calib_windows` windows are skipped - that's the calibration
        segment itself, so predicting it back would be trivial. Sequences whose
        target has SBP/DBP == NaN are also skipped - label_windows_with_bp
        (core_processing.py) masks a window NaN when it sits inside a device
        pause/dropout too far from any real vitals sample; feeding that NaN in
        as a training target would poison the loss (and gradients) for the
        whole batch instead of just being absent. The input windows themselves
        (ECG/SKNA) are still valid, so only the sequence whose LAST window is
        NaN is dropped - earlier windows in a sequence may still have NaN
        targets without disqualifying the sequence, since only positions[-1]
        is ever used as the target (see __getitem__)."""
        sequences = []
        fillna_labels = labels.copy()
        for c in group_keys:
            fillna_labels[c] = fillna_labels[c].fillna("")
        target_valid = labels["SBP"].notna().to_numpy() & labels["DBP"].notna().to_numpy()
        n_skipped = 0
        for _, group_df in fillna_labels.groupby(group_keys, sort=False):
            positions = group_df.index.to_numpy()
            # positions are already contiguous & time-ordered because of the sort above
            n = len(positions)
            for start in range(0, n - seq_len + 1, stride):
                # calib_windows=None/0 means "no calibration segment to exclude"
                # (matches the positions[:calib_windows] slice semantics used
                # elsewhere in this class, where None/0 slices to everything/nothing).
                if calib_windows and start + seq_len - 1 < calib_windows:
                    continue
                target_pos = positions[start + seq_len - 1]
                if not target_valid[target_pos]:
                    n_skipped += 1
                    continue
                sequences.append(positions[start:start + seq_len])
        if not sequences:
            raise ValueError(
                f"No sequences of length {seq_len} could be built - every recording "
                f"has fewer than {seq_len} windows, or every candidate target is "
                f"NaN (masked gap/dropout). Reduce seq_len or check bp_labels.csv."
            )
        return np.stack(sequences, axis=0)

    def _fit_norm_stats(self):
        """Compute per-channel mean/std over every window used in ANY sequence in this dataset."""
        idx = np.unique(self.seq_index.reshape(-1))
        a = self._arrays
        def stat(x):
            m, s = float(x[idx].mean()), float(x[idx].std())
            return m, (s if s > 1e-8 else 1.0)
        # Optional branches (fd/cwt) only get stats if their arrays are present.
        self.norm_stats = {k: stat(a[k]) for k in self.NORM_CHANNELS if k in a}

    def subset(self, seq_indices, fit_norm=False, norm_stats=None):
        """
        Build a new dataset sharing the same underlying arrays but restricted to
        the given sequence indices (e.g. a LOSO train/test split).
        - fit_norm=True: compute fresh normalization stats from this subset only
          (use this for the TRAIN subset).
        - norm_stats=<dict>: reuse stats computed elsewhere (use this for the
          TEST/VAL subset, passing the TRAIN subset's stats, to avoid leakage).
        """
        if fit_norm and norm_stats is not None:
            raise ValueError("Pass either fit_norm=True or norm_stats, not both.")
        new = SKNASequenceDataset.__new__(SKNASequenceDataset)
        new._arrays = self._arrays
        new.seq_index = self.seq_index[np.asarray(seq_indices)]
        new.seq_len = self.seq_len
        new.calib_windows = self.calib_windows
        new.norm_mode = self.norm_mode
        # Arrays are shared and already decimated, so subsets inherit the factor
        # for record-keeping only - they never re-decimate.
        new.downsample = getattr(self, "downsample", 1)
        new.norm_stats = None if fit_norm else (norm_stats or self.norm_stats)
        if fit_norm:
            new._fit_norm_stats()
        return new

    def split_within_subjects(self, subjects, val_frac=0.15, group_cols=("Subject_ID", "Session_ID")):
        """
        Split sequences belonging to `subjects` into train/val pools by TIME
        within each recording (not by holding out a whole subject): the first
        (1 - val_frac) of each recording's windows are eligible for train
        sequences, the last val_frac for val sequences. Sequences whose
        windows straddle that cutoff are dropped entirely (a small number),
        which guarantees zero window overlap between the two pools - no
        leakage via overlapping sliding-window sequences.

        Use this instead of holding out one whole subject as "the" validation
        subject: with only a handful of subjects, picking one specific
        subject for validation makes early stopping hostage to however
        atypical that one subject happens to be (e.g. an unusually wide BP
        range subject can make validation loss look bad/noisy regardless of
        whether the model actually generalizes). Splitting by time within
        every remaining subject gives a validation signal that reflects the
        whole training population instead of one subject's idiosyncrasies,
        and doesn't sacrifice an entire subject's data just to validate.

        Returns (train_seq_idx, val_seq_idx): integer positions into this
        dataset's sequences (i.e. valid indices for .subset()).
        """
        labels = self._arrays["labels"]
        n = len(labels)
        keys = [c for c in group_cols if c in labels.columns]
        fillna_labels = labels.copy()
        for c in keys:
            fillna_labels[c] = fillna_labels[c].fillna("")

        is_val_position = np.zeros(n, dtype=bool)
        for _, group_df in fillna_labels.groupby(keys, sort=False):
            positions = group_df.index.to_numpy()   # time-sorted & contiguous (see constructor)
            cutoff = int(len(positions) * (1 - val_frac))
            is_val_position[positions[cutoff:]] = True

        subjects_set = set(subjects)
        last_pos = self.seq_index[:, -1]
        seq_subject = labels["Subject_ID"].to_numpy()[last_pos]
        relevant = np.isin(seq_subject, list(subjects_set))

        pos_is_val = is_val_position[self.seq_index]      # (num_sequences, seq_len)
        all_val = pos_is_val.all(axis=1)
        all_train = ~pos_is_val.any(axis=1)

        train_idx = np.where(relevant & all_train)[0]
        val_idx = np.where(relevant & all_val)[0]
        return train_idx, val_idx

    def sequence_groups(self, group_col="Subject_ID"):
        """Subject_ID (or other group_col) for each sequence - for LOSO splitting."""
        labels = self._arrays["labels"]
        last_pos = self.seq_index[:, -1]
        return labels[group_col].to_numpy()[last_pos]

    def __len__(self):
        return len(self.seq_index)

    def __getitem__(self, idx):
        positions = self.seq_index[idx]           # (seq_len,) row-positions, time order
        target_pos = positions[-1]                  # predict BP at the last (most recent) window
        a = self._arrays
        labels = a["labels"]
        ns = self.norm_stats

        def norm(x, key):
            if self.norm_mode == "calib":
                # Every position in a sequence belongs to the same recording
                # (sequences never cross a group boundary), so any row's stats
                # describe the whole sequence - read them off the target window.
                mean_col, std_col = a["calib_norm"][key]
                m, s = mean_col[target_pos], std_col[target_pos]
            else:
                m, s = ns[key]
            return (x - m) / s

        ecg_seq = norm(a["ecg"][positions], "ecg")        # (seq_len, window_size)
        skna_seq = norm(a["skna"][positions], "skna")
        rskna_seq = norm(a["rskna"][positions], "rskna")
        iskna_seq = norm(a["iskna"][positions], "iskna")
        askna_seq = norm(a["askna"][positions], "askna")   # (seq_len,)

        item = {
            # (seq_len, 1, window_size): channel dim added for Conv1d, applied per-timestep
            "ecg": torch.from_numpy(ecg_seq).float().unsqueeze(1),
            "skna": torch.from_numpy(skna_seq).float().unsqueeze(1),
            "rskna": torch.from_numpy(rskna_seq).float().unsqueeze(1),
            "iskna": torch.from_numpy(iskna_seq).float().unsqueeze(1),
            "askna": torch.from_numpy(askna_seq).float().unsqueeze(1),   # (seq_len, 1)
            "sbp": torch.tensor(labels["SBP"].iloc[target_pos], dtype=torch.float32),
            "dbp": torch.tensor(labels["DBP"].iloc[target_pos], dtype=torch.float32),
            # per-recording calibration baseline (raw mmHg) - see _compute_baselines
            "baseline_sbp": torch.tensor(a["baseline_sbp"][target_pos], dtype=torch.float32),
            "baseline_dbp": torch.tensor(a["baseline_dbp"][target_pos], dtype=torch.float32),
            "subject_id": labels["Subject_ID"].iloc[target_pos],
        }
        # Optional branches, only when present. FD already carries its own
        # 2-channel dim (FFT, PSD) so no unsqueeze; CWT is a single-channel
        # image, so add a channel dim for Conv2d.
        if "fd" in a:
            fd_seq = norm(a["fd"][positions], "fd")             # (seq_len, 2, n_bins)
            item["fd"] = torch.from_numpy(fd_seq).float()
        if "cwt" in a:
            cwt_seq = norm(a["cwt"][positions], "cwt")          # (seq_len, n_freq, n_time)
            item["cwt"] = torch.from_numpy(cwt_seq).float().unsqueeze(1)  # (seq_len, 1, F, T)
        return item


def make_loso_splits(dataset, group_col="Subject_ID"):
    """
    Yields (train_seq_idx, test_seq_idx, held_out_subject) for leave-one-subject-out
    CV over the given dataset's sequences. Splits are at the SEQUENCE level but
    grouped by subject, so a subject with multiple sessions (e.g. s5) is always
    held out as one unit - no session-level leakage.
    """
    groups = dataset.sequence_groups(group_col)
    for subject in sorted(set(groups)):
        test_idx = np.where(groups == subject)[0]
        train_idx = np.where(groups != subject)[0]
        yield train_idx, test_idx, subject


def three_way_time_split(dataset, subject, adapt_end=0.5, val_end=0.6,
                         group_cols=("Subject_ID", "Session_ID")):
    """Split ONE subject's own sequences into three temporally-disjoint buckets,
    for per-subject fine-tuning (personalization / hybrid calibration):

        adapt : windows in the first `adapt_end` of the recording  (calibration)
        val   : windows in [adapt_end, val_end)                    (early-stopping)
        test  : windows in [val_end, 1.0]                          (held-out eval)

    A sequence counts for a bucket only if ALL of its windows fall inside that
    bucket, so sequences straddling a boundary are dropped - guaranteeing zero
    window overlap between adapt/val/test (no sliding-window leakage). Fractions
    are computed per (Subject_ID, Session_ID) so multi-session subjects don't
    leak across sessions.

    Fallback for short recordings: if the narrow middle val band contains no full
    sequence (val empty) but adapt has >=2 seqs and test is non-empty, the val
    slice is carved from the TAIL of the adapt band (its latest ~15% by time, so
    still before test). Subjects whose native val band is non-empty are unaffected.

    Returns (adapt_idx, val_idx, test_idx): integer positions into dataset's
    sequences (valid indices for dataset.subset()).
    """
    labels = dataset._arrays["labels"]
    n = len(labels)
    fillna = labels.copy()
    keys = [c for c in group_cols if c in fillna.columns]
    for c in keys:
        fillna[c] = fillna[c].fillna("")

    frac = np.full(n, -1.0)   # within-recording position of each row in [0, 1)
    for _, g in fillna.groupby(keys, sort=False):
        pos = g.index.to_numpy()             # time-sorted & contiguous
        if len(pos):
            frac[pos] = np.arange(len(pos)) / len(pos)

    seq_frac = frac[dataset.seq_index]                       # (num_seq, seq_len)
    last_pos = dataset.seq_index[:, -1]
    seq_subject = labels["Subject_ID"].to_numpy()[last_pos]
    mine = seq_subject == subject

    in_adapt = (seq_frac < adapt_end).all(axis=1)
    in_val = ((seq_frac >= adapt_end) & (seq_frac < val_end)).all(axis=1)
    in_test = (seq_frac >= val_end).all(axis=1)

    adapt_idx = np.where(mine & in_adapt)[0]
    val_idx = np.where(mine & in_val)[0]
    test_idx = np.where(mine & in_test)[0]

    if len(val_idx) == 0 and len(adapt_idx) >= 2 and len(test_idx) > 0:
        order = np.argsort(last_pos[adapt_idx])              # earliest -> latest
        adapt_sorted = adapt_idx[order]
        n_val = max(1, int(round(len(adapt_sorted) * 0.15)))
        val_idx = adapt_sorted[-n_val:]
        adapt_idx = adapt_sorted[:-n_val]
    return adapt_idx, val_idx, test_idx