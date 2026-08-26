from PyQt5.QtCore import QThread, pyqtSignal
import os
import numpy as np
import pandas as pd
import traceback

import core_processing as core


class PipelineWorker(QThread):
    progress = pyqtSignal(str, int, int)
    stage_started = pyqtSignal(str)
    stage_finished = pyqtSignal(str, dict)
    log = pyqtSignal(str)
    error = pyqtSignal(str)
    finished_all = pyqtSignal(dict)

    def __init__(self, params, parent=None):
        super().__init__(parent)
        self.params = params
        self._stop_requested = False
        self.results = {}

    def request_stop(self):
        self._stop_requested = True

    def _should_stop(self):
        return self._stop_requested

    def run(self):
        try:
            p = self.params
            out_dir = p.get("output_dir") or "."
            fs = p["fs"]
            window_sec = p.get("window_sec", 5)
            # Whole-recording cubic detrend of raw SKNA before bandpass.
            # Set to False (here or via params) to A/B its effect on features.
            detrend_skna = p.get("detrend_skna", False)

            subject_id = (p.get("subject_id") or "").strip()
            session_id = (p.get("session_id") or "").strip()
            recording_tag = "_".join(x for x in (subject_id, session_id) if x) or "recording"
            recording_dir = os.path.join(out_dir, recording_tag)

            labels_dir = os.path.join(recording_dir, p.get("labels_subdir", "windowing_labeling"))
            features_dir = os.path.join(recording_dir, p.get("features_subdir", "features"))
            os.makedirs(labels_dir, exist_ok=True)
            os.makedirs(features_dir, exist_ok=True)

            self.stage_started.emit("load")
            self.log.emit(f"Recording: {recording_tag}")
            self.log.emit(f"Output dirs -> labels/windows: {labels_dir} | features: {features_dir}")
            self.log.emit("Loading TXT and CSV files...")
            df_txt = core.load_txt_signal(p["txt_filename"])
            df_csv = core.load_csv_vitals(p["csv_filename"])
            df_txt, df_csv, align_msg = core.align_time_axes(
                df_txt, df_csv, fs,
                method=p.get("align_method", "hr_xcorr"),
                align_offset_sec=p.get("align_offset_sec"),
            )
            self.log.emit(f"Alignment: {align_msg}")

            if self._should_stop():
                return self._finish_cancelled()

            skna_raw = pd.read_csv(
                p["txt_filename"], skiprows=14, usecols=[3], encoding='latin1'
            ).values.flatten()

            # --- optional high-pass filter of the RAW signal (very first step) --
            # A Butterworth HPF (default fc=0.08 Hz, order 2) removes slow
            # baseline wander / DC drift from the raw channels before anything
            # else. Its main effect is on the ECG (the `raw_signal` dataset) and
            # the raw-signal plots; the SKNA path's 500-1000 Hz band-pass already
            # removes everything below 500 Hz, so the HPF is a near-no-op there.
            # Applied to the full-length signal (before clipping) so the filter's
            # long edge transients land at the true recording ends.
            hpf_enabled = bool(p.get("hpf_enabled", False))
            hpf_fc = float(p.get("hpf_fc", 0.08))
            hpf_order = int(p.get("hpf_order", 2))
            if hpf_enabled:
                self.log.emit(f"High-pass filtering raw signal "
                               f"(Butterworth order {hpf_order}, fc={hpf_fc:g} Hz)...")
                skna_raw = core.highpass_filter(skna_raw, fs, fc=hpf_fc, order=hpf_order)
                df_txt = core.highpass_channels(
                    df_txt, channels=('CH1', 'CH40', 'CH41'), fs=fs,
                    fc=hpf_fc, order=hpf_order)

            self.results["df_txt"] = df_txt
            self.results["df_csv"] = df_csv
            self.stage_finished.emit("load", {"df_txt": df_txt, "df_csv": df_csv})

            if self._should_stop():
                return self._finish_cancelled()

            self.stage_started.emit("detrend")
            detrend_full_signal = p.get("detrend_full_signal", True)
            detrend_full_type = p.get("detrend_full_type", "linear")
            if detrend_full_signal:
                self.log.emit(f"Detrending full-length CH1/CH40/CH41 ({detrend_full_type})...")
                df_txt = core.detrend_channels(
                    df_txt, channels=('CH1', 'CH40', 'CH41'), type=detrend_full_type)
                self.results["df_txt"] = df_txt
            else:
                self.log.emit("Full-signal detrending skipped.")
            self.stage_finished.emit("detrend", {"df_txt": df_txt})

            if self._should_stop():
                return self._finish_cancelled()

            self.stage_started.emit("preprocess")
            self.log.emit(f"Preprocessing SKNA at fs={fs} Hz "
                           f"({len(skna_raw)} samples, {len(skna_raw)/fs/60:.1f} min)...")
            skna_filtered, rskna, iskna, askna = core.preprocess_skna(
                skna_raw, fs, window_sec=window_sec, detrend=detrend_skna)

            if self._should_stop():
                return self._finish_cancelled()

            self.results.update(dict(skna_filtered=skna_filtered, rskna=rskna,
                                      iskna=iskna, askna=askna, fs=fs))
            self.stage_finished.emit("preprocess", {
                "skna_filtered": skna_filtered, "rskna": rskna,
                "iskna": iskna, "askna": askna,
            })

            self.stage_started.emit("window")
            skna_windows, rskna_windows, iskna_windows, window_size = core.rectWindow(
                fs=fs, window_sec=window_sec, skna_filtered=skna_filtered, rskna=rskna, iskna=iskna
            )
            self.log.emit(f"Created {len(skna_windows)} windows of {window_sec}s each.")
            self.results.update(dict(skna_windows=skna_windows, rskna_windows=rskna_windows,
                                      iskna_windows=iskna_windows, window_size=window_size))
            self.stage_finished.emit("window", {
                "skna_windows": skna_windows, "rskna_windows": rskna_windows,
                "iskna_windows": iskna_windows, "window_size": window_size,
            })

            if self._should_stop():
                return self._finish_cancelled()

            self.stage_started.emit("label")
            self.log.emit("Labeling windows with ground-truth SBP/DBP...")
            bp_labels = core.label_windows_with_bp(
                df_csv, window_sec=window_sec, num_windows=len(skna_windows),
                output_filename=f"{labels_dir}/bp_labels.csv",
                # Default = one window length: a window whose centre sits farther
                # than that from any real vitals sample is inside a device
                # pause/dropout, so it's masked (NaN) instead of np.interp silently
                # bridging the gap. No GUI control exists for bp_max_gap_sec yet,
                # so leaving the old default=None meant this safety check never
                # actually ran. Explicit 0/False in p still disables it.
                max_gap_sec=p.get("bp_max_gap_sec", window_sec))
            n_hr = bp_labels.attrs.get("n_hr_dropped", 0)
            n_gap = bp_labels.attrs.get("n_gap_masked", 0)
            if n_hr or n_gap:
                self.log.emit(f"Vitals cleaning: dropped {n_hr} HR==0 warm-up/invalid "
                               f"rows before interpolation"
                               + (f"; masked {n_gap} windows inside device gaps (SBP/DBP=NaN)"
                                  if n_gap else "") + ".")
            self.results["bp_labels"] = bp_labels
            self.log.emit(f"Labeled {len(bp_labels)} windows "
                           f"(SBP mean={bp_labels['SBP'].mean():.1f}, "
                           f"DBP mean={bp_labels['DBP'].mean():.1f}).")
            self.stage_finished.emit("label", {"bp_labels": bp_labels})

            self.log.emit("Windowing ECG and saving the 4 labeled training datasets...")
            ecg_col = "CH1_detrended" if (detrend_full_signal and "CH1_detrended" in df_txt.columns) else "CH1"
            self.log.emit(f"Using '{ecg_col}' as the ECG source for windowing.")
            ecg_raw = df_txt[ecg_col].to_numpy()
            ecg_windows = core.window_raw_signal(ecg_raw, fs=fs, window_sec=window_sec)

            detrend_signals = p.get("detrend_signals", True)
            detrend_type = p.get("detrend_type", "linear")
            self.log.emit(f"Detrending each signal ({detrend_type}) before export: "
                           f"{'on' if detrend_signals else 'off'}.")

            output_format = p.get("output_format", "npz")
            if output_format == "npz":
                self.log.emit("Saving labeled datasets as compact .npz (float32) instead of wide CSV.")
                labels_df, labeled_arrays = core.save_labeled_datasets_npz(
                    labels_dir, bp_labels,
                    ecg_windows=ecg_windows,
                    skna_windows=skna_windows,
                    rskna_windows=rskna_windows,
                    iskna_windows=iskna_windows,
                    askna=askna,
                    detrend=detrend_signals,
                    detrend_type=detrend_type,
                    subject_id=subject_id or None,
                    session_id=session_id or None,
                )
                self.results["labeled_datasets"] = labeled_arrays
                self.results["labels_df"] = labels_df
                for name in labeled_arrays:
                    self.log.emit(f"Saved {name}.npz + labels.csv to {labels_dir}: "
                                   f"{len(labels_df)} labeled windows.")
            else:
                labeled_datasets = core.save_labeled_datasets(
                    labels_dir, bp_labels,
                    ecg_windows=ecg_windows,
                    skna_windows=skna_windows,
                    rskna_windows=rskna_windows,
                    iskna_windows=iskna_windows,
                    askna=askna,
                    detrend=detrend_signals,
                    detrend_type=detrend_type,
                    subject_id=subject_id or None,
                    session_id=session_id or None,
                )
                self.results["labeled_datasets"] = labeled_datasets
                for name, df in labeled_datasets.items():
                    self.log.emit(f"Saved {name}_with_labels.csv to {labels_dir}: "
                                   f"{df.shape[0]} rows x {df.shape[1]} cols.")

            if self._should_stop():
                return self._finish_cancelled()

            self.stage_started.emit("features")

            def make_cb(name):
                def cb(cur, total):
                    self.progress.emit(name, cur, total)
                return cb

            time_feats = core.extract_all_time_features(
                skna_windows, rskna_windows, iskna_windows,
                f"{features_dir}/time_features.csv",
                progress_cb=make_cb("time_features"), should_stop=self._should_stop)
            if self._should_stop():
                return self._finish_cancelled()

            fft_feats = core.extract_fft_features(
                skna_windows, fs, f"{features_dir}/fft_features.csv",
                progress_cb=make_cb("fft_features"), should_stop=self._should_stop)
            if self._should_stop():
                return self._finish_cancelled()

            psd_feats = core.extract_psd_features(
                skna_windows, fs, f"{features_dir}/psd_features.csv",
                progress_cb=make_cb("psd_features"), should_stop=self._should_stop)
            if self._should_stop():
                return self._finish_cancelled()

            fftband_feats = core.extract_fftband_features(
                skna_windows, fs, f"{features_dir}/fftband_features.csv",
                progress_cb=make_cb("fftband_features"), should_stop=self._should_stop)
            if self._should_stop():
                return self._finish_cancelled()

            psdband_feats = core.extract_psdband_features(
                skna_windows, fs, f"{features_dir}/psdband_features.csv",
                progress_cb=make_cb("psdband_features"), should_stop=self._should_stop)
            if self._should_stop():
                return self._finish_cancelled()

            delta_feats = core.extract_phase_delta_features(
                fftband_feats, psdband_feats, window_sec, f"{features_dir}/delta_features.csv")
            recovery_feats = core.extract_recovery_features(
                fftband_feats, psdband_feats, window_sec, f"{features_dir}/recovery_features.csv")
            top_freq_feats = core.extract_top_frequency_features(
                fftband_feats, window_sec, f"{features_dir}/top_freq_features.csv")

            askna_feat = pd.DataFrame([{
                "aSKNA_mean": np.mean(askna), "aSKNA_std": np.std(askna),
                "aSKNA_min": np.min(askna), "aSKNA_max": np.max(askna),
                "aSKNA_range": np.max(askna) - np.min(askna), "aSKNA_median": np.median(askna),
                "aSKNA_iqr": np.percentile(askna, 75) - np.percentile(askna, 25),
            }])
            askna_feat.to_csv(f"{features_dir}/askna_features.csv", index=False)

            feature_payload = {
                "time_features": time_feats, "fft_features": fft_feats, "psd_features": psd_feats,
                "fftband_features": fftband_feats, "psdband_features": psdband_feats,
                "delta_features": delta_feats, "recovery_features": recovery_feats,
                "top_freq_features": top_freq_feats, "askna_features": askna_feat,
            }
            self.results.update(feature_payload)
            self.stage_finished.emit("features", feature_payload)

            if self._should_stop():
                return self._finish_cancelled()

            if p.get("run_cwt", False):
                self.stage_started.emit("cwt")
                cwt_window_sec = p.get("cwt_window_sec", 600)
                cwt_signal_windows, _, _, _ = core.rectWindow(
                    fs=fs, window_sec=cwt_window_sec, skna_filtered=skna_filtered)
                image_folder = f"{features_dir}/cwt_images" if p.get("save_cwt_images", False) else None
                cwt_feats = core.extract_cwt_features(
                    cwt_signal_windows, fs, f"{features_dir}/cwt_features.csv", image_folder,
                    progress_cb=make_cb("cwt_features"), should_stop=self._should_stop,
                    save_images=p.get("save_cwt_images", False))
                self.results["cwt_features"] = cwt_feats
                self.stage_finished.emit("cwt", {"cwt_features": cwt_feats})

            if self._should_stop():
                return self._finish_cancelled()

            if p.get("compute_cwt_arrays", False) or p.get("compute_fd_arrays", False):
                core.write_labels_csv(f"{labels_dir}/labels.csv", bp_labels, len(skna_windows),
                                       subject_id=subject_id or None, session_id=session_id or None)

            if p.get("compute_cwt_arrays", False):
                self.stage_started.emit("cwt_arrays")
                cwt_time_bins = p.get("cwt_time_bins", 128)
                self.log.emit(f"Computing per-window CWT arrays for training "
                               f"({len(skna_windows)} x {window_sec}s windows, "
                               f"downsampled to {cwt_time_bins} time bins)...")
                cwt_arrays = core.extract_cwt_arrays(
                    skna_windows, fs, f"{labels_dir}/cwt_signal.npz",
                    time_bins=cwt_time_bins,
                    progress_cb=make_cb("cwt_arrays"), should_stop=self._should_stop)
                self.results["cwt_arrays"] = cwt_arrays
                if not self._should_stop() and len(cwt_arrays) != len(skna_windows):
                    self.log.emit(f"[warning] CWT array rows ({len(cwt_arrays)}) != windows/labels "
                                   f"({len(skna_windows)}); this recording's cwt_signal.npz is "
                                   f"misaligned with labels.csv and should not be aggregated as-is.")
                self.log.emit(f"Saved cwt_signal.npz to {labels_dir}: shape {cwt_arrays.shape} (uint8), "
                               f"labeled by labels.csv ({len(skna_windows)} windows).")
                self.stage_finished.emit("cwt_arrays", {"cwt_arrays": cwt_arrays})

            if self._should_stop():
                return self._finish_cancelled()

            if p.get("compute_fd_arrays", False):
                self.stage_started.emit("fd_arrays")
                self.log.emit(f"Computing per-window FD (FFT+PSD) arrays for training "
                               f"({len(skna_windows)} x {window_sec}s windows, "
                               f"500-999 Hz passband, 2 channels)...")
                fd_arrays = core.extract_fd_arrays(
                    skna_windows, fs, f"{labels_dir}/fd_signal.npz",
                    progress_cb=make_cb("fd_arrays"), should_stop=self._should_stop)
                self.results["fd_arrays"] = fd_arrays
                if not self._should_stop() and len(fd_arrays) != len(skna_windows):
                    self.log.emit(f"[warning] FD array rows ({len(fd_arrays)}) != windows/labels "
                                   f"({len(skna_windows)}); this recording's fd_signal.npz is "
                                   f"misaligned with labels.csv and should not be aggregated as-is.")
                self.log.emit(f"Saved fd_signal.npz to {labels_dir}: shape {fd_arrays.shape} (float32), "
                               f"labeled by labels.csv ({len(skna_windows)} windows).")
                self.stage_finished.emit("fd_arrays", {"fd_arrays": fd_arrays})

            if self._should_stop():
                return self._finish_cancelled()

            self.stage_started.emit("phase_stats")
            time_askna_min = (np.arange(len(askna)) * window_sec) / 60.0
            df_askna = pd.DataFrame({'Time_min': time_askna_min, 'aSKNA': askna})

            vital_cols = [c for c in df_csv.columns
                          if any(t in c.upper() for t in ['SYS', 'DIA', 'MAP'])]
            df_vitals = df_csv.copy()

            askna_stats, _ = core.calculate_phase_stats(df_askna, 'Time_min', ['aSKNA'])
            vitals_stats, _ = core.calculate_phase_stats(df_vitals, 'Time_min', vital_cols)

            phase_payload = {
                "df_askna": df_askna, "df_vitals": df_vitals,
                "askna_stats": askna_stats, "vitals_stats": vitals_stats,
            }
            self.results.update(phase_payload)
            self.stage_finished.emit("phase_stats", phase_payload)

            self.log.emit("Pipeline complete.")
            self.finished_all.emit(self.results)

        except Exception as e:
            tb = traceback.format_exc()
            self.error.emit(f"{e}\n\n{tb}")

    def _finish_cancelled(self):
        self.log.emit("Pipeline cancelled.")
        self.error.emit("__CANCELLED__")