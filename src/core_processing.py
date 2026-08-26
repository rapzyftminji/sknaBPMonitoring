import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # safe default; GUI embeds figures via FigureCanvasQTAgg instead
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib import image as mpimg
import pywt

from scipy.fft import rfft, rfftfreq
from scipy.signal import welch
from scipy import signal

def load_txt_signal(txt_file, skiprows=14, usecols=(0, 1, 2, 3),
                     names=('Time_ms', 'CH1', 'CH40', 'CH41'), encoding='latin1'):
    df_txt = pd.read_csv(
        txt_file, skiprows=skiprows, usecols=list(usecols),
        names=list(names), encoding=encoding
    )
    return df_txt


def load_csv_vitals(csv_file):
    return pd.read_csv(csv_file)


def _ecg_to_hr(ecg, fs, grid_sec=1.0):
    """Heart-rate series (t_sec, hr_bpm on a uniform grid) from raw ECG, for time
    alignment. Pan-Tompkins-lite: bandpass 5-18 Hz to isolate QRS, differentiate
    + square + moving-average, then peaks >=0.3 s apart above an adaptive height.
    High-rate ECG (10 kHz) is decimated to ~1 kHz first purely for speed."""
    ecg = np.nan_to_num(np.asarray(ecg, dtype=np.float64))
    if fs > 2000:
        q = int(round(fs / 1000.0))
        ecg = signal.decimate(ecg, q, ftype="fir", zero_phase=True)
        fs = fs / q
    nyq = fs / 2.0
    b, a = signal.butter(2, [5.0 / nyq, min(18.0, nyq * 0.95) / nyq], btype="band")
    x = signal.filtfilt(b, a, ecg)
    dx = np.diff(x, prepend=x[:1])
    win = max(1, int(0.10 * fs))
    integ = np.convolve(dx * dx, np.ones(win) / win, mode="same")
    thr = np.median(integ) + 0.5 * (np.percentile(integ, 95) - np.median(integ))
    peaks, _ = signal.find_peaks(integ, distance=int(0.3 * fs), height=thr)
    if len(peaks) < 3:
        return None, None
    tb = peaks / fs
    hr = 60.0 / np.diff(tb)
    ok = (hr > 30) & (hr < 200)               # drop detector glitches
    t_hr, hr = tb[1:][ok], hr[ok]
    if len(t_hr) < 3:
        return None, None
    grid = np.arange(0.0, tb[-1], grid_sec)
    return grid, np.interp(grid, t_hr, hr)


def _hr_xcorr_lag(t_e, hr_e, t_d, hr_d, max_lag_sec=300.0, grid=1.0):
    """Lag L (s) maximising Pearson corr of the two demeaned HR series, defined
    so that device_time = ecg_time + L. Returns (L, peak_r)."""
    g = np.arange(0.0, min(t_e[-1], t_d[-1]), grid)
    if len(g) < 30:
        return None, None
    e = np.interp(g, t_e, hr_e)
    e = e - e.mean()
    best_l, best_c = 0.0, -2.0
    for L in np.arange(-max_lag_sec, max_lag_sec + grid, grid):
        d = np.interp(g + L, t_d, hr_d, left=np.nan, right=np.nan)
        m = ~np.isnan(d)
        if m.sum() < 30:
            continue
        ee, dd = e[m] - e[m].mean(), d[m] - d[m].mean()
        den = np.sqrt((ee * ee).sum() * (dd * dd).sum())
        if den < 1e-9:
            continue
        c = float((ee * dd).sum() / den)
        if c > best_c:
            best_c, best_l = c, float(L)
    return best_l, best_c


def _vitals_rel_seconds(df_csv):
    """Elapsed seconds from the first vitals row. Prefers the epoch-ms TimeStamp
    column (precise, monotonic, survives midnight); falls back to the time-of-day
    column, then the row index. Used for BOTH the HR-xcorr lag and the Time_min
    axis so the two always share the same clock."""
    ts_col = _find_vital_col(df_csv, ['TIMESTAMP'])
    if ts_col is not None:
        return (df_csv[ts_col] - df_csv[ts_col].iloc[0]).to_numpy(dtype=float) / 1000.0
    tcol = next((c for c in df_csv.columns
                 if 'time' in c.lower() or ':' in str(df_csv[c].iloc[0])), None)
    if tcol is not None:
        wall = pd.to_datetime(df_csv[tcol], errors="coerce", format="mixed")
        return (wall - wall.iloc[0]).dt.total_seconds().to_numpy(dtype=float)
    return df_csv.index.to_numpy(dtype=float)


def _vitals_time_and_hr(df_csv):
    """(t_rel_sec, hr_bpm, hr_col) from a vitals frame - shared relative-time
    clock plus the device's own HeartRate column (None if absent)."""
    hr_col = _find_vital_col(df_csv, ['HEARTRATE', 'HEART RATE', 'PULSE'])
    t = _vitals_rel_seconds(df_csv)
    hr = df_csv[hr_col].to_numpy(dtype=float) if hr_col else None
    return t, hr, hr_col


def estimate_bp_alignment_offset(df_txt, df_csv, fs, ecg_col="CH1", max_lag_sec=300.0):
    """Time offset L (seconds) between the ECG/SKNA .txt and the BP-device vitals,
    measured by cross-correlating ECG-derived HR against the device's own
    HeartRate column: device_time = ecg_time + L. Clock-independent, so it works
    even when the two instruments' clocks are minutes apart. Returns (L, peak_r),
    or (None, None) if HR can't be recovered from either side. The standalone
    audit in src/diagnostics_alignment.py motivated this (see alignment_audit.csv)."""
    if ecg_col not in df_txt.columns:
        return None, None
    t_d, hr_d, _ = _vitals_time_and_hr(df_csv)
    if hr_d is None:
        return None, None
    valid = hr_d > 0                              # drop the device's warm-up rows
    if valid.sum() < 5:
        return None, None
    t_e, hr_e = _ecg_to_hr(df_txt[ecg_col].to_numpy(), fs)
    if t_e is None:
        return None, None
    return _hr_xcorr_lag(t_e, hr_e, t_d[valid], hr_d[valid], max_lag_sec)


def align_time_axes(df_txt, df_csv, fs, method="hr_xcorr", ecg_col="CH1",
                    align_offset_sec=None, min_xcorr_r=0.30, max_lag_sec=300.0):
    """Put the ECG/SKNA .txt and the BP-device vitals on ONE time axis.

    df_txt time = sample index / fs (starts at 0). df_csv time = its own
    timestamps (starts at 0). They are then shifted onto the ECG clock:

    method="hr_xcorr" (default): the true start offset is measured by
        cross-correlating ECG-derived HR against the device HeartRate column
        (see estimate_bp_alignment_offset). This is the fix for the old
        end-matching, which - because the two instruments start/stop at
        different wall-clock times and their clocks can be minutes apart -
        mislabelled every window by ~65-200 s (src/diagnostics_alignment.py).
    method="end": legacy fallback - assume both streams stopped at the same
        instant and match their END times. Used automatically when the HR match
        is too weak (peak r < min_xcorr_r) or HR is unavailable.

    align_offset_sec: manual override in seconds (device_time = ecg_time +
        offset); bypasses the HR cross-correlation for one recording.

    Sample index -> minutes uses the true fs, so a 10 kHz recording is not
    stretched 5x the way the old hard-coded 0.5 ms/sample assumption did.
    """
    df_txt = df_txt.copy()
    df_csv = df_csv.copy()
    df_txt['Time_min'] = df_txt.index / (fs * 60.0)
    df_csv['Time_min'] = _vitals_rel_seconds(df_csv) / 60.0

    lag, r, src = None, None, None
    if align_offset_sec is not None:
        lag, src = float(align_offset_sec), "manual"
    elif method == "hr_xcorr":
        lag, r = estimate_bp_alignment_offset(df_txt, df_csv, fs, ecg_col, max_lag_sec)
        src = "hr_xcorr"

    trust = align_offset_sec is not None or (lag is not None and r is not None and r >= min_xcorr_r)
    if lag is not None and trust:
        # device_time = ecg_time + lag  =>  express BP on the ECG clock.
        df_csv['Time_min'] -= lag / 60.0
        rtxt = "" if r is None else f", r={r:.2f}"
        msg = (f"Aligned by {src}: BP shifted {-lag:+.1f}s onto the ECG clock "
               f"(offset device=ecg{lag:+.1f}s{rtxt}).")
    else:
        max_txt, max_csv = df_txt['Time_min'].max(), df_csv['Time_min'].max()
        if max_txt > max_csv:
            df_txt['Time_min'] -= (max_txt - max_csv)
        elif max_csv > max_txt:
            df_csv['Time_min'] -= (max_csv - max_txt)
        why = ("HR match too weak" if method == "hr_xcorr" else f"method={method}")
        rtxt = "" if r is None else f" (best r={r:.2f} < {min_xcorr_r})"
        msg = (f"Aligned by END-MATCH fallback ({why}{rtxt}) - WARNING: assumes "
               f"both streams stopped together; often wrong by tens of seconds.")

    return df_txt, df_csv, msg


def normalize_minmax(arr, axis=None, eps=1e-8):
    """
    Rescale arr to [0, 1] via min-max normalization (along axis, or globally
    if axis=None). A near-constant signal (max - min < eps) is mapped to all
    zeros instead of dividing by ~0.
    """
    arr = np.asarray(arr, dtype=float)
    keepdims = axis is not None
    lo = np.min(arr, axis=axis, keepdims=keepdims)
    hi = np.max(arr, axis=axis, keepdims=keepdims)
    rng = hi - lo
    rng = np.where(rng < eps, eps, rng)
    return (arr - lo) / rng


def normalize_log_minmax(arr, log_eps=1e-12, floor_percentile=1.0, ceil_percentile=99.0):
    """
    Log-compress (log10(arr + log_eps)) then rescale to [0, 1] using ROBUST
    (percentile-based) endpoints instead of the exact min/max.

    Spectral power/magnitude spans many orders of magnitude bin-to-bin - a
    plain linear normalize_minmax lets one or two dominant bins crush every
    other bin toward 0, destroying the spectrum's shape. Taking the log first
    fixes that, but using the exact min/max is still fragile: a single
    unrepresentative bin (e.g. a numerical null deep in a filter's stopband)
    can set the "0" reference far below the spectrum's real noise floor. That
    then makes genuinely-attenuated content (a properly filtered-out band)
    read as a deceptively HIGH value instead of near 0, since it ends up much
    closer to the (real) peak than to that one outlier bin. Clipping to the
    [floor_percentile, ceil_percentile] range before scaling avoids that:
    values at/below the floor percentile map to 0, at/above the ceiling
    percentile map to 1, everything else scales linearly between them.
    """
    log_vals = np.log10(np.asarray(arr, dtype=float) + log_eps)
    lo = np.percentile(log_vals, floor_percentile)
    hi = np.percentile(log_vals, ceil_percentile)
    rng = hi - lo
    if rng < 1e-8:
        rng = 1e-8
    return np.clip((log_vals - lo) / rng, 0.0, 1.0)


def detrend_full_signal(arr, type='linear'):
    """
    Detrend one entire continuous 1D signal (e.g. a full CH1/CH40/CH41 recording),
    as opposed to detrend_signal() below which works per-window. Use this BEFORE
    windowing, so slow baseline wander across the whole recording is removed
    instead of being re-fit independently inside every 5s window.

    type: 'linear' (remove best-fit line over the whole recording) or
          'constant' (just remove the overall mean).
    """
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"detrend_full_signal expects 1D input, got shape {arr.shape}")
    return signal.detrend(arr, type=type)


def detrend_channels(df_txt, channels=('CH1', 'CH40', 'CH41'), type='linear'):
    """
    Detrend the given channels over the full recording length, then rescale
    each detrended channel to [0, 1] (see normalize_minmax) so every channel
    ends up on the same, comparable, unitless scale.

    Returns a COPY of df_txt with new '<channel>_detrended' columns added
    alongside the original raw columns (which are left untouched), so both
    can be plotted/compared side by side. Any missing channel is skipped.
    """
    df = df_txt.copy()
    for ch in channels:
        if ch in df.columns:
            detrended = detrend_full_signal(df[ch].to_numpy(), type=type)
            df[f"{ch}_detrended"] = normalize_minmax(detrended)
    return df


def highpass_filter(x, fs, fc=0.08, order=2):
    """
    Zero-phase Butterworth high-pass filter, applied to a 1D signal.

    Defaults (fc=0.08 Hz, order=2) target slow baseline wander / DC drift - the
    standard cleanup for the raw ECG that becomes the `raw_signal` dataset.
    fc is very low relative to fs, so the filter is built as second-order
    sections (numerically stable at tiny normalized cutoffs, unlike b/a form)
    and applied with sosfiltfilt for zero phase shift (no ECG feature is moved
    in time). Returns float64. A signal too short for the filter's edge padding
    is returned unchanged.
    """
    x = np.asarray(x, dtype=np.float64)
    nyq = fs / 2.0
    wn = fc / nyq
    if not (0.0 < wn < 1.0):
        raise ValueError(f"high-pass fc={fc} Hz is out of range for fs={fs} Hz "
                         f"(normalized cutoff {wn:.4g} must be in (0, 1)).")
    sos = signal.butter(order, wn, btype='high', output='sos')
    # sosfiltfilt needs len(x) > 3*(2*order) padding; guard tiny inputs.
    if len(x) <= 6 * order + 1:
        return x
    return signal.sosfiltfilt(sos, x)


def highpass_channels(df_txt, channels=('CH1', 'CH40', 'CH41'), fs=10000, fc=0.08, order=2):
    """
    Apply highpass_filter() IN PLACE to the raw channel columns of df_txt (a
    copy), so downstream detrending/windowing/plots all see the de-drifted
    signal. Any missing channel is skipped.
    """
    df = df_txt.copy()
    for ch in channels:
        if ch in df.columns:
            df[ch] = highpass_filter(df[ch].to_numpy(), fs, fc=fc, order=order)
    return df


def preprocess_skna(skna_raw, fs, window_sec=5, detrend=True):
    skna = skna_raw * 1000.0

    if detrend:
        time = np.arange(len(skna)) / fs
        p = np.polyfit(time, skna, deg=3)
        trend = np.polyval(p, time)
        skna_detrended = (skna - trend) + np.mean(skna)
    else:
        # Skip the whole-recording cubic detrend. NOTE: the 500-1000 Hz
        # bandpass below already removes low-frequency baseline wander, so
        # this flag usually changes the features only slightly - flip it and
        # re-measure corr(aSKNA, SBP) rather than assuming an effect.
        skna_detrended = skna
    nyquist = fs / 2.0
    b, a = signal.butter(3, [500.0 / nyquist, 999.0 / nyquist], btype='bandpass')
    skna_filtered = signal.filtfilt(b, a, skna_detrended)

    rskna = np.abs(skna_filtered)

    dt = 1.0 / fs
    time_constant = 0.01
    alpha = dt / (time_constant + dt)
    b_iir = [alpha]
    a_iir = [1.0, -(1.0 - alpha)]
    iskna = signal.lfilter(b_iir, a_iir, rskna)

    # Must match rectWindow()/label_windows_with_bp()'s window_sec, or askna
    # ends up with a different row count than skna_windows/labels/CWT/FD
    # arrays - skna_dataset.py's row-count guard will reject the mismatch.
    window_size = int(np.round(window_sec * fs))
    num_windows = len(rskna) // window_size
    askna = np.mean(rskna[:num_windows * window_size].reshape(num_windows, window_size), axis=1)

    # Each of the 4 outputs is independently rescaled to [0, 1] as a final
    # step, AFTER the detrend/filter/rectify/integrate/average computations
    # above (which run on the original scale) - so the pipeline's internal
    # math is unaffected, only the values handed back out are normalized.
    skna_filtered = normalize_minmax(skna_filtered)
    rskna = normalize_minmax(rskna)
    iskna = normalize_minmax(iskna)
    askna = normalize_minmax(askna)

    return skna_filtered, rskna, iskna, askna


def rectWindow(fs=10000, window_sec=5, skna_filtered=None, rskna=None, iskna=None):
    window_size = window_sec * fs
    num_windows = len(skna_filtered) // window_size

    skna_windows = skna_filtered[:num_windows * window_size].reshape(num_windows, window_size)

    if rskna is None and iskna is None:
        return skna_windows, None, None, window_size

    rskna_windows = rskna[:num_windows * window_size].reshape(num_windows, window_size)
    iskna_windows = iskna[:num_windows * window_size].reshape(num_windows, window_size)
    return skna_windows, rskna_windows, iskna_windows, window_size


def extract_all_time_features(skna_windows, rskna_windows, iskna_windows,
                               output_filename=None, progress_cb=None, should_stop=None):
    feature_list = []
    n = len(skna_windows)
    for i in range(n):
        if should_stop and should_stop():
            break
        w_skna, w_rskna, w_iskna = skna_windows[i], rskna_windows[i], iskna_windows[i]
        features = {
            "Window_ID": i,
            "SKNA_mean": np.mean(w_skna), "SKNA_std": np.std(w_skna), "SKNA_var": np.var(w_skna),
            "SKNA_rms": np.sqrt(np.mean(w_skna ** 2)), "SKNA_min": np.min(w_skna), "SKNA_max": np.max(w_skna),
            "SKNA_range": np.max(w_skna) - np.min(w_skna), "SKNA_median": np.median(w_skna),
            "rSKNA_mean": np.mean(w_rskna), "rSKNA_std": np.std(w_rskna), "rSKNA_var": np.var(w_rskna),
            "rSKNA_rms": np.sqrt(np.mean(w_rskna ** 2)), "rSKNA_min": np.min(w_rskna), "rSKNA_max": np.max(w_rskna),
            "rSKNA_range": np.max(w_rskna) - np.min(w_rskna), "rSKNA_median": np.median(w_rskna),
            "iSKNA_mean": np.mean(w_iskna), "iSKNA_std": np.std(w_iskna), "iSKNA_var": np.var(w_iskna),
            "iSKNA_rms": np.sqrt(np.mean(w_iskna ** 2)), "iSKNA_min": np.min(w_iskna), "iSKNA_max": np.max(w_iskna),
            "iSKNA_range": np.max(w_iskna) - np.min(w_iskna), "iSKNA_median": np.median(w_iskna),
        }
        feature_list.append(features)
        if progress_cb:
            progress_cb(i + 1, n)

    df_features = pd.DataFrame(feature_list)
    if output_filename:
        os.makedirs(os.path.dirname(output_filename) or ".", exist_ok=True)
        df_features.to_csv(output_filename, index=False)
    return df_features


def extract_fft_features(windows, fs, output_filename=None, progress_cb=None, should_stop=None):
    window_size = windows.shape[1]
    f_axis = rfftfreq(window_size, d=1 / fs)
    feature_list = []
    n = len(windows)
    for i, window in enumerate(windows):
        if should_stop and should_stop():
            break
        # Remove the window's own DC offset before the FFT. Without this, a
        # normalized (see normalize_minmax) but originally zero-mean signal
        # sits on a ~0.5 baseline, which shows up as a huge spurious 0 Hz
        # spike that swamps the real 500-999 Hz content. Then log-compress
        # before normalizing (see normalize_log_minmax) so the spectrum's
        # shape survives instead of being crushed by whichever bin is largest.
        mag_FFT = normalize_log_minmax(np.abs(rfft(window - np.mean(window))))
        features = {
            'Window_ID': i, 'FFT_total_magnitude': np.sum(mag_FFT), 'FFT_mean_magnitude': np.mean(mag_FFT),
            'FFT_std_magnitude': np.std(mag_FFT), 'FFT_max_magnitude': np.max(mag_FFT),
            'FFT_peak_frequency': f_axis[1 + np.argmax(mag_FFT[1:])],
            'FFT_median_magnitude': np.median(mag_FFT), 'FFT_p90_magnitude': np.percentile(mag_FFT, 90),
            'FFT_p95_magnitude': np.percentile(mag_FFT, 95), 'FFT_p99_magnitude': np.percentile(mag_FFT, 99),
        }
        feature_list.append(features)
        if progress_cb:
            progress_cb(i + 1, n)

    df_features = pd.DataFrame(feature_list)
    if output_filename:
        os.makedirs(os.path.dirname(output_filename) or ".", exist_ok=True)
        df_features.to_csv(output_filename, sep=',', index=False)
    return df_features


def extract_psd_features(windows, fs, output_filename=None, progress_cb=None, should_stop=None):
    window_size = windows.shape[1]
    feature_list = []
    n = len(windows)
    for i, window in enumerate(windows):
        if should_stop and should_stop():
            break
        # detrend='constant' (explicit, matches scipy's own default) removes each
        # window's DC offset before the periodogram, for the same reason as the
        # FFT path above.
        f_psd, psd = welch(window, fs, nperseg=window_size, detrend='constant')
        psd = normalize_log_minmax(psd)
        band_idx = np.logical_and(f_psd >= 500, f_psd <= 999)
        psd_band = psd[band_idx]
        features = {
            'Window_ID': i, 'PSD_total_power_500_999': np.sum(psd_band), 'PSD_mean_power': np.mean(psd_band),
            'PSD_std_power': np.std(psd_band), 'PSD_max_power': np.max(psd_band),
            'PSD_peak_frequency': f_psd[np.argmax(psd)], 'PSD_median_power': np.median(psd),
        }
        feature_list.append(features)
        if progress_cb:
            progress_cb(i + 1, n)

    df_features = pd.DataFrame(feature_list)
    if output_filename:
        os.makedirs(os.path.dirname(output_filename) or ".", exist_ok=True)
        df_features.to_csv(output_filename, sep=',', index=False)
    return df_features


def extract_fftband_features(windows, fs, output_filename=None, progress_cb=None, should_stop=None):
    window_size = windows.shape[1]
    f_axis = rfftfreq(window_size, d=1 / fs)
    feature_list = []
    n = len(windows)
    for i, window in enumerate(windows):
        if should_stop and should_stop():
            break
        fft_vals = rfft(window - np.mean(window))  # remove DC offset - see extract_fft_features
        # NOT normalized: these features are ratios (each band's energy over
        # the total), already scale-invariant - min-max normalizing first would
        # inject a shift (subtracting a min) that distorts those ratios instead
        # of leaving them invariant, so the raw magnitude is used as-is.
        mag_FFT = np.abs(fft_vals)
        fft_energy = mag_FFT ** 2
        band_total = (f_axis >= 500) & (f_axis < 1000)
        total_energy = np.sum(fft_energy[band_total])

        features = {'Window_ID': i}
        for start in range(500, 1000, 10):
            end = start + 10
            idx = (f_axis >= start) & (f_axis < end)
            band_energy = np.sum(fft_energy[idx])
            features[f"FFT_share_{start}_{end}"] = band_energy / total_energy if total_energy > 0 else 0
        idx_505 = (f_axis >= 505) & (f_axis < 515)
        features["FFT_share_505_515"] = np.sum(fft_energy[idx_505]) / total_energy if total_energy > 0 else 0

        feature_list.append(features)
        if progress_cb:
            progress_cb(i + 1, n)

    df_features = pd.DataFrame(feature_list)
    if output_filename:
        os.makedirs(os.path.dirname(output_filename) or ".", exist_ok=True)
        df_features.to_csv(output_filename, sep=',', index=False)
    return df_features


def extract_psdband_features(windows, fs, output_filename=None, progress_cb=None, should_stop=None):
    window_size = windows.shape[1]
    feature_list = []
    n = len(windows)
    for i, window in enumerate(windows):
        if should_stop and should_stop():
            break
        # NOT normalized: same reasoning as extract_fftband_features - these
        # are ratios (band power / total power), already scale-invariant.
        f_psd, psd = welch(window, fs, nperseg=window_size, detrend='constant')
        band_idx = np.logical_and(f_psd >= 500, f_psd <= 999)
        total_power = np.sum(psd[band_idx])

        features = {'Window_ID': i}
        for start in range(500, 1000, 10):
            end = start + 10
            idx = (f_psd >= start) & (f_psd < end)
            band_power = np.sum(psd[idx])
            features[f"PSD_share_{start}_{end}"] = band_power / total_power if total_power > 0 else 0
        idx_505 = (f_psd >= 505) & (f_psd < 515)
        features["PSD_share_505_515"] = np.sum(psd[idx_505]) / total_power if total_power > 0 else 0

        feature_list.append(features)
        if progress_cb:
            progress_cb(i + 1, n)

    df_features = pd.DataFrame(feature_list)
    if output_filename:
        os.makedirs(os.path.dirname(output_filename) or ".", exist_ok=True)
        df_features.to_csv(output_filename, sep=',', index=False)
    return df_features


def extract_cwt_features(windows, fs, output_filename=None, image_folder=None,
                          progress_cb=None, should_stop=None, save_images=True):
    """
    Per-window CWT scalogram. The raw wavelet magnitude has the same
    orders-of-magnitude skew as FFT/PSD, so it's log-compressed and robustly
    (percentile-based) normalized to [0, 1] via normalize_log_minmax before
    anything else - both the summary features below and any saved image are
    computed from that normalized magnitude, not the raw one.

    When save_images=True, each window's scalogram is written as a uint8
    grayscale PNG (0-255) via a direct pixel-array write (matplotlib.image.
    imsave) instead of a full matplotlib Figure/Axes/colorbar/title - this is
    far cheaper per window (no Figure objects, no text/colorbar rendering, no
    DPI upscaling), which matters a lot here since a single CWT "window" can
    be minutes of samples across dozens of scales. The same normalized
    grayscale array is also saved as a companion .npy (uint8) file, so it can
    be loaded straight back as a numeric array (e.g. as a DL model input)
    without re-decoding the PNG.
    """
    if image_folder:
        os.makedirs(image_folder, exist_ok=True)

    feature_list = []
    frequencies = np.arange(500, 999, 10)
    wavelet = "cmor5.0-1.0"
    scales = pywt.central_frequency(wavelet) * fs / frequencies

    n = len(windows)
    for i, window in enumerate(windows):
        if should_stop and should_stop():
            break
        coeffs, freqs = pywt.cwt(window, scales, wavelet, sampling_period=1 / fs)
        mag = normalize_log_minmax(np.abs(coeffs))

        features = {
            "Window_ID": i, "CWT_mean_magnitude": np.mean(mag),
            "CWT_std_magnitude": np.std(mag), "CWT_max_magnitude": np.max(mag),
        }
        feature_list.append(features)

        if save_images and image_folder:
            gray = (mag * 255).astype(np.uint8)
            mpimg.imsave(os.path.join(image_folder, f"window_{i:03d}.png"),
                         gray, cmap='gray', vmin=0, vmax=255, origin='lower')
            np.save(os.path.join(image_folder, f"window_{i:03d}.npy"), gray)

        if progress_cb:
            progress_cb(i + 1, n)

    df = pd.DataFrame(feature_list)
    if output_filename:
        os.makedirs(os.path.dirname(output_filename) or ".", exist_ok=True)
        df.to_csv(output_filename, index=False)
    return df


def compute_single_cwt(window, fs, freq_min=500, freq_max=999, freq_step=10):
    """Compute one CWT scalogram (mag + extent info) for interactive preview in the GUI,
    without touching disk."""
    frequencies = np.arange(freq_min, freq_max, freq_step)
    wavelet = "cmor5.0-1.0"
    scales = pywt.central_frequency(wavelet) * fs / frequencies
    coeffs, freqs = pywt.cwt(window, scales, wavelet, sampling_period=1 / fs)
    mag = np.abs(coeffs)
    return mag, frequencies


def _pool_columns_mean(arr2d, n_out):
    """
    Average-pool a 2D array's columns (axis=1) down to n_out columns. Bucket
    boundaries come from np.linspace so buckets stay ~even even when n_in
    isn't a multiple of n_out (e.g. 50000 samples -> 128 bins).
    """
    n_in = arr2d.shape[1]
    if n_out >= n_in:
        return arr2d.astype(float)
    edges = np.linspace(0, n_in, n_out + 1).astype(int)
    pooled = np.empty((arr2d.shape[0], n_out), dtype=float)
    for j in range(n_out):
        start, end = edges[j], max(edges[j + 1], edges[j] + 1)
        pooled[:, j] = arr2d[:, start:end].mean(axis=1)
    return pooled


def compute_cwt_array(window, fs, freq_min=500, freq_max=999, freq_step=10, time_bins=128):
    """
    Compute one window's CWT scalogram as a compact, DL-ready grayscale array:
    log-compressed + robustly normalized to [0, 1] (see normalize_log_minmax,
    same treatment as FFT/PSD), then average-pooled along time down to
    `time_bins` columns, then quantized to uint8 (0-255).

    The time-axis pooling is a deliberate size/storage tradeoff, not a
    correctness fix - a complex Morlet wavelet's own time-domain width
    already smooths detail far below the raw sample spacing, so pooling
    mostly discards redundant oversampling rather than real structure, as
    long as time_bins stays coarser than the wavelet's native resolution but
    finer than a typical burst's duration. See extract_cwt_arrays() for
    batch use across many windows (e.g. all 3s training windows).
    """
    frequencies = np.arange(freq_min, freq_max, freq_step)
    wavelet = "cmor5.0-1.0"
    scales = pywt.central_frequency(wavelet) * fs / frequencies
    coeffs, _ = pywt.cwt(window, scales, wavelet, sampling_period=1 / fs)
    mag = normalize_log_minmax(np.abs(coeffs))
    mag = _pool_columns_mean(mag, time_bins)
    return (mag * 255).astype(np.uint8)


def extract_cwt_arrays(windows, fs, output_path=None, freq_min=500, freq_max=999,
                        freq_step=10, time_bins=128, progress_cb=None, should_stop=None):
    """
    Compute the compact DL-ready CWT array (see compute_cwt_array) for every
    window and stack them into one (num_windows, n_freq, time_bins) uint8
    array, saved as a single compressed npz (key 'X') if output_path is
    given - matching how ECG/SKNA/rSKNA/iSKNA/aSKNA are already saved (one
    file per recording), rather than one file per window like
    extract_cwt_features()'s exploratory image dump.
    """
    n = len(windows)
    arrays = []
    stopped_early = False
    for i, window in enumerate(windows):
        if should_stop and should_stop():
            stopped_early = True
            break
        arrays.append(compute_cwt_array(window, fs, freq_min, freq_max, freq_step, time_bins))
        if progress_cb:
            progress_cb(i + 1, n)

    n_freq = len(np.arange(freq_min, freq_max, freq_step))
    stacked = np.stack(arrays, axis=0) if arrays else np.empty((0, n_freq, time_bins), dtype=np.uint8)
    # A cancelled run produces fewer arrays than windows/labels. Persisting that
    # partial stack would leave a cwt_signal.npz silently misaligned with the
    # already-written labels.csv, which then corrupts aggregation (every later
    # recording's rows get shifted out of sync). So only write a complete stack.
    if output_path and not stopped_early:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        np.savez_compressed(output_path, X=stacked)
    return stacked


def _fd_band_slice(window_size, fs, freq_min, freq_max):
    """Index mask selecting the [freq_min, freq_max] Hz bins of a length-
    window_size rfft / Welch spectrum. rfft(N) and welch(nperseg=N) share the
    exact same frequency grid (df = fs/N), so one mask serves both - which is
    also why FFT magnitude and PSD can be stacked as aligned 2-channel input."""
    freqs = rfftfreq(window_size, d=1 / fs)
    return (freqs >= freq_min) & (freqs <= freq_max)


def compute_fd_array(window, fs, band_mask=None, freq_min=500, freq_max=999):
    """
    Compute one 5s window's frequency-domain representation as a compact,
    DL-ready 2-channel array: channel 0 = FFT magnitude, channel 1 = Welch
    PSD, each restricted to the [freq_min, freq_max] Hz passband, then
    rescaled to [0, 1] via plain linear min-max (see normalize_minmax).

    NOTE: this is a deliberate departure from normalize_log_minmax (used
    elsewhere for FFT/PSD summary features and the CWT) at the user's
    explicit request (2026-07-17), despite spectral power/magnitude spanning
    orders of magnitude bin-to-bin - a single dominant bin can crush every
    other bin toward 0 under plain linear min-max. See
    [[feedback_spectral_normalization]] in memory for the original reasoning
    this overrides.

    The passband slice is taken BEFORE normalizing so the [0, 1] range
    reflects real passband structure instead of being anchored by the
    near-zero stopband bins the 500-999 Hz bandpass already removed. band_mask
    (from _fd_band_slice) can be precomputed once and passed in to avoid
    recomputing the frequency grid for every window.

    Returns a (2, n_band_bins) float32 array. See extract_fd_arrays() for
    batch use across all of a recording's 5s windows.
    """
    window_size = len(window)
    if band_mask is None:
        band_mask = _fd_band_slice(window_size, fs, freq_min, freq_max)
    w = window - np.mean(window)   # DC removal, see extract_fft_features

    mag = np.abs(rfft(w))[band_mask]
    f_psd, psd = welch(w, fs, nperseg=window_size, detrend='constant')
    psd = psd[band_mask]

    fft_norm = normalize_minmax(mag)
    psd_norm = normalize_minmax(psd)
    return np.stack([fft_norm, psd_norm], axis=0).astype(np.float32)


def extract_fd_arrays(windows, fs, output_path=None, freq_min=500, freq_max=999,
                       progress_cb=None, should_stop=None):
    """
    Compute the compact DL-ready FD array (see compute_fd_array) for every
    window and stack them into one (num_windows, 2, n_band_bins) float32 array
    - channel 0 FFT magnitude, channel 1 Welch PSD - saved as a single
    compressed npz (key 'X') if output_path is given, matching how the other
    per-3s-window training signals are saved (one file per recording).
    """
    window_size = windows.shape[1] if len(windows) else 0
    band_mask = _fd_band_slice(window_size, fs, freq_min, freq_max) if window_size else None
    n_bins = int(band_mask.sum()) if band_mask is not None else 0

    n = len(windows)
    arrays = []
    stopped_early = False
    for i, window in enumerate(windows):
        if should_stop and should_stop():
            stopped_early = True
            break
        arrays.append(compute_fd_array(window, fs, band_mask=band_mask,
                                        freq_min=freq_min, freq_max=freq_max))
        if progress_cb:
            progress_cb(i + 1, n)

    stacked = np.stack(arrays, axis=0) if arrays else np.empty((0, 2, n_bins), dtype=np.float32)
    # See extract_cwt_arrays: don't persist a partial stack from a cancelled run,
    # or fd_signal.npz ends up misaligned with labels.csv and corrupts aggregation.
    if output_path and not stopped_early:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        np.savez_compressed(output_path, X=stacked)
    return stacked


def _phase_labels(window_ids, window_sec):
    time_min = (window_ids * window_sec) / 60.0
    cycle_time = time_min % 10
    conditions = [(cycle_time < 4.5), (cycle_time >= 4.5) & (cycle_time < 5.5), (cycle_time >= 5.5)]
    labels = ['Before', 'During', 'After']
    return np.select(conditions, labels, default='After')


def extract_phase_delta_features(df_fft, df_psd, window_sec=5, output_filename=None):
    df_combined = pd.concat([df_fft, df_psd.drop(columns=['Window_ID'], errors='ignore')], axis=1)
    df_combined['Phase'] = _phase_labels(df_fft['Window_ID'].to_numpy(), window_sec)

    mean_before = df_combined[df_combined['Phase'] == 'Before'].drop(columns=['Window_ID', 'Phase']).mean()
    mean_during = df_combined[df_combined['Phase'] == 'During'].drop(columns=['Window_ID', 'Phase']).mean()
    delta = mean_during - mean_before

    results = {}
    fft_cols = [c for c in delta.index if 'FFT_share' in c and c != 'FFT_share_505_515']
    if fft_cols:
        results['FFT_delta_during_before_505_515'] = delta.get('FFT_share_505_515', 0)
        abs_fft = delta[fft_cols].abs().dropna()
        if not abs_fft.empty:
            results['FFT_delta_abs_max_band'] = abs_fft.idxmax().replace('FFT_share_', '')
            results['FFT_delta_abs_max_value'] = delta[abs_fft.idxmax()]
        else:
            results['FFT_delta_abs_max_band'] = "None"
            results['FFT_delta_abs_max_value'] = 0
        results['FFT_delta_positive_band_count'] = (delta[fft_cols] > 0).sum()
        results['FFT_delta_negative_band_count'] = (delta[fft_cols] < 0).sum()

    psd_cols = [c for c in delta.index if 'PSD_share' in c and c != 'PSD_share_505_515']
    if psd_cols:
        results['PSD_delta_during_before_505_515'] = delta.get('PSD_share_505_515', 0)
        abs_psd = delta[psd_cols].abs().dropna()
        if not abs_psd.empty:
            results['PSD_delta_abs_max_band'] = abs_psd.idxmax().replace('PSD_share_', '')
            results['PSD_delta_abs_max_value'] = delta[abs_psd.idxmax()]
        else:
            results['PSD_delta_abs_max_band'] = "None"
            results['PSD_delta_abs_max_value'] = 0
        results['PSD_delta_positive_band_count'] = (delta[psd_cols] > 0).sum()
        results['PSD_delta_negative_band_count'] = (delta[psd_cols] < 0).sum()

    df_results = pd.DataFrame([results])
    if output_filename:
        os.makedirs(os.path.dirname(output_filename) or ".", exist_ok=True)
        df_results.to_csv(output_filename, index=False)
    return df_results

def extract_recovery_features(df_fft, df_psd, window_sec=5, output_filename=None):
    df_combined = pd.concat([df_fft, df_psd.drop(columns=['Window_ID'], errors='ignore')], axis=1)
    df_combined['Phase'] = _phase_labels(df_fft['Window_ID'].to_numpy(), window_sec)

    mean_during = df_combined[df_combined['Phase'] == 'During'].drop(columns=['Window_ID', 'Phase']).mean()
    mean_after = df_combined[df_combined['Phase'] == 'After'].drop(columns=['Window_ID', 'Phase']).mean()
    rel_change = (mean_after - mean_during) / mean_during.replace(0, np.nan)

    results = {}
    fft_cols = [c for c in rel_change.index if 'FFT_share' in c and c != 'FFT_share_505_515']
    if fft_cols:
        results['FFT_rel_after_during_505_515'] = rel_change.get('FFT_share_505_515', 0)
        fft_drops = rel_change[fft_cols].dropna()
        if not fft_drops.empty and fft_drops.min() < 0:
            results['FFT_rel_drop_max_band'] = fft_drops.idxmin().replace('FFT_share_', '')
            results['FFT_rel_drop_max_value'] = fft_drops.min()
        else:
            results['FFT_rel_drop_max_band'] = "None"
            results['FFT_rel_drop_max_value'] = 0
        results['FFT_rel_increase_band_count'] = (rel_change[fft_cols] > 0).sum()
        results['FFT_rel_decrease_band_count'] = (rel_change[fft_cols] < 0).sum()

    psd_cols = [c for c in rel_change.index if 'PSD_share' in c and c != 'PSD_share_505_515']
    if psd_cols:
        results['PSD_rel_after_during_505_515'] = rel_change.get('PSD_share_505_515', 0)
        psd_drops = rel_change[psd_cols].dropna()
        if not psd_drops.empty and psd_drops.min() < 0:
            results['PSD_rel_drop_max_band'] = psd_drops.idxmin().replace('PSD_share_', '')
            results['PSD_rel_drop_max_value'] = psd_drops.min()
        else:
            results['PSD_rel_drop_max_band'] = "None"
            results['PSD_rel_drop_max_value'] = 0
        results['PSD_rel_increase_band_count'] = (rel_change[psd_cols] > 0).sum()
        results['PSD_rel_decrease_band_count'] = (rel_change[psd_cols] < 0).sum()

    df_results = pd.DataFrame([results])
    if output_filename:
        os.makedirs(os.path.dirname(output_filename) or ".", exist_ok=True)
        df_results.to_csv(output_filename, index=False)
    return df_results

def extract_top_frequency_features(df_fft, window_sec=5, output_filename=None):
    df_combined = df_fft.copy()
    df_combined['Phase'] = _phase_labels(df_fft['Window_ID'].to_numpy(), window_sec)

    mean_during = df_combined[df_combined['Phase'] == 'During'].drop(columns=['Window_ID', 'Phase']).mean()
    band_cols = [c for c in mean_during.index if 'FFT_share' in c and c != 'FFT_share_505_515']
    top_bands = mean_during[band_cols].sort_values(ascending=False)

    results = {}
    for rank in range(1, 7):
        if rank <= len(top_bands):
            band_name = top_bands.index[rank - 1]
            share_val = top_bands.iloc[rank - 1]
            parts = band_name.split('_')
            start_freq, end_freq = int(parts[2]), int(parts[3])
            results[f'FFT_during_top{rank}_band_center'] = (start_freq + end_freq) / 2.0
            results[f'FFT_during_top{rank}_share'] = share_val
        else:
            results[f'FFT_during_top{rank}_band_center'] = 0
            results[f'FFT_during_top{rank}_share'] = 0

    df_results = pd.DataFrame([results])
    if output_filename:
        os.makedirs(os.path.dirname(output_filename) or ".", exist_ok=True)
        df_results.to_csv(output_filename, index=False)
    return df_results


def _find_vital_col(df_csv, keywords):
    for col in df_csv.columns:
        if any(k.upper() in col.upper() for k in keywords):
            return col
    return None


def label_windows_with_bp(df_csv, window_sec, num_windows, time_col='Time_min',
                           sbp_col=None, dbp_col=None, samples_per_window=10,
                           output_filename=None, drop_invalid_hr=True,
                           max_gap_sec=None):
    """
    Label each fixed-length signal window with ground-truth SBP/DBP.

    For window i, the covered interval is:
        [i * window_sec / 60, (i + 1) * window_sec / 60)  (minutes)

    Because vitals (SBP/DBP) are usually sampled at a coarser rate than the
    3s SKNA windows, we linearly interpolate the vitals signal and evaluate
    it at `samples_per_window` evenly spaced points inside each window, then
    average those interpolated values. This avoids empty-window issues that
    a plain "average the raw rows falling in this window" approach would hit
    whenever the vitals sampling interval is longer than window_sec.

    Parameters
    ----------
    df_csv : DataFrame with a time column (minutes) and vitals columns.
    window_sec : window length in seconds (should match rectWindow's window_sec).
    num_windows : number of signal windows (e.g. len(skna_windows)).
    sbp_col, dbp_col : explicit column names; auto-detected if None.
    samples_per_window : number of interpolation points averaged per window.
    output_filename : optional CSV path to save the resulting labels.
    drop_invalid_hr : drop vitals rows with HeartRate==0 (device warm-up / not
        locked) before interpolating, so their held pre-calibration SBP/DBP does
        not anchor the first ~20-30 s of labels. On by default; no-op if no HR
        column is present.
    max_gap_sec : if set, windows whose centre is farther than this from any real
        vitals sample sit inside a device pause/dropout and get SBP/DBP=NaN
        instead of a value np.interp would fabricate across the gap. Off (None)
        by default, since NaN targets require the downstream dataset builder to
        drop them. n_hr_dropped / n_gap_masked are recorded on df.attrs.

    Returns
    -------
    DataFrame with columns: Window_ID, Time_start_min, Time_end_min, SBP, DBP
    """
    if sbp_col is None:
        sbp_col = _find_vital_col(df_csv, ['SYS', 'SYSTOLIC', 'SBP'])
    if dbp_col is None:
        dbp_col = _find_vital_col(df_csv, ['DIA', 'DIAS', 'DIASTOLIC', 'DBP'])

    if sbp_col is None or dbp_col is None:
        raise ValueError(
            f"Could not auto-detect SBP/DBP columns in df_csv. "
            f"Found columns: {list(df_csv.columns)}. "
            f"Pass sbp_col/dbp_col explicitly."
        )

    # Drop the device's warm-up / not-locked rows (HeartRate==0): those carry a
    # held, pre-calibration SBP/DBP that is not a real reading, and would anchor
    # the interpolation at a bogus value for the first ~20-30 s. Uses the HR
    # column when present; a no-op otherwise.
    keep_cols = [time_col, sbp_col, dbp_col]
    hr_col = _find_vital_col(df_csv, ['HEARTRATE', 'HEART RATE', 'PULSE']) if drop_invalid_hr else None
    if hr_col and hr_col not in keep_cols:
        keep_cols = keep_cols + [hr_col]
    df_sorted = df_csv[keep_cols].dropna(subset=[time_col, sbp_col, dbp_col]).sort_values(time_col)
    if hr_col:
        n0 = len(df_sorted)
        df_sorted = df_sorted[df_sorted[hr_col] > 0]
        n_dropped = n0 - len(df_sorted)
    else:
        n_dropped = 0
    t = df_sorted[time_col].to_numpy()
    sbp_vals_all = df_sorted[sbp_col].to_numpy()
    dbp_vals_all = df_sorted[dbp_col].to_numpy()

    if len(t) < 2:
        raise ValueError("Need at least 2 valid vitals samples to interpolate BP labels.")

    window_min = window_sec / 60.0
    # A window whose centre is farther than max_gap_sec from any real vitals
    # sample sits inside a device pause/dropout; np.interp would silently bridge
    # it into a fabricated label, so mark it NaN (invalid) instead.
    max_gap_min = (max_gap_sec / 60.0) if max_gap_sec else None
    records = []
    n_gap_masked = 0
    for i in range(num_windows):
        start = i * window_min
        end = start + window_min
        query_times = np.linspace(start, end, samples_per_window, endpoint=False)
        # np.interp clamps outside [t.min(), t.max()] to the edge values,
        # which is a reasonable fallback for windows slightly outside vitals range.
        sbp_interp = np.interp(query_times, t, sbp_vals_all)
        dbp_interp = np.interp(query_times, t, dbp_vals_all)
        if max_gap_min is not None:
            centre = start + window_min / 2.0
            if np.min(np.abs(t - centre)) > max_gap_min:
                sbp_interp = dbp_interp = np.array([np.nan])
                n_gap_masked += 1
        records.append({
            "Window_ID": i,
            "Time_start_min": start,
            "Time_end_min": end,
            "SBP": np.mean(sbp_interp),
            "DBP": np.mean(dbp_interp),
        })

    df_labels = pd.DataFrame(records)
    # Non-breaking bookkeeping so callers (worker) can log what was cleaned.
    df_labels.attrs["n_hr_dropped"] = int(n_dropped)
    df_labels.attrs["n_gap_masked"] = int(n_gap_masked)
    if output_filename:
        os.makedirs(os.path.dirname(output_filename) or ".", exist_ok=True)
        df_labels.to_csv(output_filename, index=False)
    return df_labels


def window_raw_signal(raw_signal, fs, window_sec):
    """
    Segment a 1D raw signal (e.g. raw ECG) into fixed-length windows the same
    way rectWindow() does for SKNA, so window i lines up with the same time
    span as skna_windows[i] / bp_labels row i (as long as raw_signal and the
    SKNA source have the same length and sampling rate).
    """
    window_size = int(round(window_sec * fs))
    num_windows = len(raw_signal) // window_size
    return raw_signal[:num_windows * window_size].reshape(num_windows, window_size)


def detrend_signal(arr, type='linear'):
    """
    Remove trend from a signal before labeling/export.
      - 2D input (num_windows, window_size): detrends each window independently
        along axis=1, so one window's trend can't leak into another.
      - 1D input (num_windows,), e.g. aSKNA: detrends the whole series over time.
    type: 'linear' (remove best-fit line) or 'constant' (just remove the mean).
    """
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 2:
        return signal.detrend(arr, axis=1, type=type)
    elif arr.ndim == 1:
        return signal.detrend(arr, type=type)
    else:
        raise ValueError(f"detrend_signal expects 1D or 2D input, got shape {arr.shape}")


def _build_labeled_csv(bp_labels, output_filename, num_windows, detrend=True,
                        detrend_type='linear', subject_id=None, session_id=None,
                        **named_signals):
    """
    Build one CSV: Subject_ID, Session_ID (if given), Window_ID, SBP, DBP placed
    first (so they're immediately visible), followed by the given named signal(s).
    Each named signal can be:
      - a 1D array of length >= num_windows (one scalar per window, e.g. aSKNA), or
      - a 2D array of shape (>=num_windows, window_size) (per-sample columns,
        named '<signal_name>_0', '<signal_name>_1', ...).

    If detrend=True (default), each signal is detrended via detrend_signal()
    before being written out.

    subject_id / session_id: optional identifiers stamped onto every row. These
    are what later let you concatenate CSVs from many recordings into one
    training set and still know which rows came from which subject (needed for
    subject-wise / leave-one-subject-out splitting).
    """
    df = pd.DataFrame({"Window_ID": np.arange(num_windows)})
    df = df.merge(bp_labels[["Window_ID", "SBP", "DBP"]], on="Window_ID", how="left")

    missing = df["SBP"].isna().sum()
    if missing:
        print(f"[warning] {missing}/{num_windows} windows in {output_filename} "
              f"have no matching SBP/DBP label (Window_ID mismatch or out-of-range time).")

    for name, arr in named_signals.items():
        arr = np.asarray(arr)[:num_windows]
        if detrend:
            # detrend_signal can push values outside [0, 1] again (it re-fits
            # a trend per window/series on already-normalized data), so
            # re-normalize afterward to guarantee the saved values are [0, 1].
            arr = normalize_minmax(detrend_signal(arr, type=detrend_type), axis=1 if arr.ndim == 2 else None)
        if arr.ndim == 1:
            df[name] = arr
        else:
            cols = pd.DataFrame(arr, columns=[f"{name}_{j}" for j in range(arr.shape[1])])
            df = pd.concat([df.reset_index(drop=True), cols.reset_index(drop=True)], axis=1)

    if session_id is not None:
        df.insert(0, "Session_ID", session_id)
    if subject_id is not None:
        df.insert(0, "Subject_ID", subject_id)
    df = df.copy()  # de-fragment after inserting on a wide (many-column) frame

    if output_filename:
        os.makedirs(os.path.dirname(output_filename) or ".", exist_ok=True)
        df.to_csv(output_filename, index=False)
    return df


def _build_labeled_npz(bp_labels, base_path, num_windows, detrend=True,
                        detrend_type='linear', subject_id=None, session_id=None,
                        **named_signals):
    """
    Compact alternative to _build_labeled_csv(). Instead of writing thousands
    of per-sample text columns (ECG_0..ECG_N) into one CSV, this writes:
      - '<base_path>_labels.csv': lightweight metadata only
        (Subject_ID, Session_ID, Window_ID, SBP, DBP) — small, human-readable,
        safe to open/inspect/diff.
      - '<base_path>_signals.npz': the actual windowed samples as
        gzip-compressed float32 binary arrays — this is where the CSV export
        was spending most of its disk space and load time.
    Returns (labels_df, signals_dict) so the GUI can still preview labels
    without pulling the (larger) signal arrays into memory.
    """
    labels_df = pd.DataFrame({"Window_ID": np.arange(num_windows)})
    labels_df = labels_df.merge(bp_labels[["Window_ID", "SBP", "DBP"]], on="Window_ID", how="left")

    missing = labels_df["SBP"].isna().sum()
    if missing:
        print(f"[warning] {missing}/{num_windows} windows in {base_path} "
              f"have no matching SBP/DBP label (Window_ID mismatch or out-of-range time).")

    signals_dict = {}
    for name, arr in named_signals.items():
        arr = np.asarray(arr, dtype=np.float32)[:num_windows]
        if detrend:
            # Re-normalize after detrending - see _build_labeled_csv for why.
            arr = normalize_minmax(
                detrend_signal(arr, type=detrend_type), axis=1 if arr.ndim == 2 else None
            ).astype(np.float32)
        signals_dict[name] = arr

    if session_id is not None:
        labels_df.insert(0, "Session_ID", session_id)
    if subject_id is not None:
        labels_df.insert(0, "Subject_ID", subject_id)
    labels_df = labels_df.copy()

    if base_path:
        os.makedirs(os.path.dirname(base_path) or ".", exist_ok=True)
        labels_df.to_csv(f"{base_path}_labels.csv", index=False)
        np.savez_compressed(f"{base_path}_signals.npz", **signals_dict)

    return labels_df, signals_dict


def save_labeled_datasets_npz(out_dir, bp_labels, ecg_windows=None, skna_windows=None,
                               rskna_windows=None, iskna_windows=None, askna=None,
                               detrend=True, detrend_type='linear',
                               subject_id=None, session_id=None):
    """
    Compact alternative to save_labeled_datasets(). Same 4 dataset kinds, but
    each is saved as a small '<name>_labels.csv' + a compressed
    '<name>_signals.npz' instead of one wide CSV. Use this when disk size /
    load time of the CSV export becomes a problem (which it does quickly once
    you have several full-length recordings at fs=10000Hz).

    Returns a dict of {name: labels_df} — the signal arrays themselves are
    NOT kept in memory here; load them back on demand with load_labeled_npz().
    """
    num_windows = len(bp_labels)
    outputs = {}
    kwargs = dict(detrend=detrend, detrend_type=detrend_type,
                  subject_id=subject_id, session_id=session_id)

    if ecg_windows is not None:
        labels_df, _ = _build_labeled_npz(
            bp_labels, f"{out_dir}/raw_signal", num_windows, ECG=ecg_windows, **kwargs)
        outputs["raw_signal"] = labels_df

    if skna_windows is not None:
        labels_df, _ = _build_labeled_npz(
            bp_labels, f"{out_dir}/skna_signal", num_windows, SKNA=skna_windows, **kwargs)
        outputs["skna_signal"] = labels_df

    if rskna_windows is not None and iskna_windows is not None:
        labels_df, _ = _build_labeled_npz(
            bp_labels, f"{out_dir}/iskna_rskna", num_windows,
            rSKNA=rskna_windows, iSKNA=iskna_windows, **kwargs)
        outputs["iskna_rskna"] = labels_df

    if askna is not None:
        labels_df, _ = _build_labeled_npz(
            bp_labels, f"{out_dir}/askna", num_windows, aSKNA=askna, **kwargs)
        outputs["askna"] = labels_df

    return outputs


def build_window_labels(bp_labels, num_windows, subject_id=None, session_id=None):
    """
    Build the per-window label table shared by every signal array of a
    recording: one row per window, columns Window_ID, SBP, DBP (+ Subject_ID/
    Session_ID when given). Every '*_signal.npz' array (including the optional
    FD/CWT arrays) is row-aligned to this table by Window_ID, so this one table
    labels all of them - there is intentionally NOT a separate labels file per
    signal kind.
    """
    labels_df = pd.DataFrame({"Window_ID": np.arange(num_windows)})
    labels_df = labels_df.merge(bp_labels[["Window_ID", "SBP", "DBP"]], on="Window_ID", how="left")
    if session_id is not None:
        labels_df.insert(0, "Session_ID", session_id)
    if subject_id is not None:
        labels_df.insert(0, "Subject_ID", subject_id)
    return labels_df


def write_labels_csv(out_path, bp_labels, num_windows, subject_id=None, session_id=None):
    """Write the shared per-window label table (see build_window_labels) to
    out_path. Safe to call more than once for a recording - it's idempotent
    (same rows every time), so a stage that needs labels.csv to exist can just
    ensure it rather than depending on another stage having written it."""
    labels_df = build_window_labels(bp_labels, num_windows, subject_id, session_id)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    labels_df.to_csv(out_path, index=False)
    return labels_df


def load_labeled_npz(base_path):
    """
    Load one dataset saved by save_labeled_datasets_npz(). base_path is the
    same prefix used at save time (e.g. '.../raw_signal'); '_labels.csv' and
    '_signals.npz' are appended automatically.

    Returns (labels_df, signals_dict): signals_dict maps signal name -> a
    float32 ndarray of shape (num_windows, window_size) or (num_windows,).
    """
    labels_df = pd.read_csv(f"{base_path}_labels.csv")
    with np.load(f"{base_path}_signals.npz") as npz:
        signals_dict = {k: npz[k] for k in npz.files}
    return labels_df, signals_dict


def save_labeled_datasets(out_dir, bp_labels, ecg_windows=None, skna_windows=None,
                           rskna_windows=None, iskna_windows=None, askna=None,
                           detrend=True, detrend_type='linear',
                           subject_id=None, session_id=None):
    """
    Save the 4 CNN-LSTM training inputs as separate, readable CSVs, each with
    SBP/DBP as the leading columns:

      1. raw_signal_with_labels.csv    -> ECG_0..ECG_{W-1}, SBP, DBP
      2. skna_signal_with_labels.csv   -> SKNA_0..SKNA_{W-1}, SBP, DBP
      3. iskna_rskna_with_labels.csv   -> rSKNA_0.., iSKNA_0.., SBP, DBP
      4. askna_with_labels.csv         -> aSKNA (one value per window), SBP, DBP

    num_windows is taken from bp_labels so all 4 files share the same
    Window_ID indexing and row count.

    detrend=True (default) removes a linear trend from every signal before
    saving: per-window for the windowed signals (ECG/SKNA/rSKNA/iSKNA), and
    across the full window sequence for aSKNA. Set detrend=False to skip, or
    detrend_type='constant' to only remove the mean instead of the slope.

    subject_id / session_id: optional identifiers stamped onto every row of
    every output CSV (see _build_labeled_csv). Pass these when processing one
    recording out of a multi-subject dataset so the per-recording CSVs can
    later be concatenated into one training set via aggregate_datasets.py
    without losing track of which subject each row belongs to.

    Returns a dict of the 4 DataFrames, keyed by name.
    """
    num_windows = len(bp_labels)
    outputs = {}

    if ecg_windows is not None:
        outputs["raw_signal"] = _build_labeled_csv(
            bp_labels, f"{out_dir}/raw_signal_with_labels.csv", num_windows,
            detrend=detrend, detrend_type=detrend_type,
            subject_id=subject_id, session_id=session_id, ECG=ecg_windows)

    if skna_windows is not None:
        outputs["skna_signal"] = _build_labeled_csv(
            bp_labels, f"{out_dir}/skna_signal_with_labels.csv", num_windows,
            detrend=detrend, detrend_type=detrend_type,
            subject_id=subject_id, session_id=session_id, SKNA=skna_windows)

    if rskna_windows is not None and iskna_windows is not None:
        outputs["iskna_rskna"] = _build_labeled_csv(
            bp_labels, f"{out_dir}/iskna_rskna_with_labels.csv", num_windows,
            detrend=detrend, detrend_type=detrend_type,
            subject_id=subject_id, session_id=session_id,
            rSKNA=rskna_windows, iSKNA=iskna_windows)

    if askna is not None:
        outputs["askna"] = _build_labeled_csv(
            bp_labels, f"{out_dir}/askna_with_labels.csv", num_windows,
            detrend=detrend, detrend_type=detrend_type,
            subject_id=subject_id, session_id=session_id, aSKNA=askna)

    return outputs


def save_labeled_datasets_npz(out_dir, bp_labels, ecg_windows=None, skna_windows=None,
                               rskna_windows=None, iskna_windows=None, askna=None,
                               detrend=True, detrend_type='linear',
                               subject_id=None, session_id=None, dtype='float32'):
    """
    Compact alternative to save_labeled_datasets(). A wide CSV with one column
    per sample (e.g. ECG_0..ECG_49999 for a 5s @ 10000Hz window) is very
    inefficient: every float is stored as ~15-20 ASCII characters instead of
    4 bytes, and text can't be compressed as well as binary. This instead
    saves:

        out_dir/labels.csv        -> Window_ID, SBP, DBP, Subject_ID, Session_ID
        out_dir/raw_signal.npz    -> {'X': (num_windows, window_size) float32}
        out_dir/skna_signal.npz   -> {'X': (num_windows, window_size) float32}
        out_dir/iskna_rskna.npz   -> {'rSKNA': ..., 'iSKNA': ...}
        out_dir/askna.npz         -> {'X': (num_windows,) float32}

    Same detrending behavior as save_labeled_datasets(). Typically 5-10x
    smaller on disk than the CSV version, and loads much faster since NumPy
    reads the binary array directly instead of parsing text.

    Returns (labels_df, dict of {kind: ndarray or {sub-key: ndarray}}).
    """
    num_windows = len(bp_labels)
    os.makedirs(out_dir, exist_ok=True)

    labels_df = write_labels_csv(f"{out_dir}/labels.csv", bp_labels, num_windows,
                                  subject_id=subject_id, session_id=session_id)

    def _prep(arr):
        arr = np.asarray(arr)[:num_windows]
        if detrend:
            # Re-normalize after detrending - see _build_labeled_csv for why.
            arr = normalize_minmax(detrend_signal(arr, type=detrend_type), axis=1 if arr.ndim == 2 else None)
        return arr.astype(dtype)

    outputs = {}
    if ecg_windows is not None:
        outputs["raw_signal"] = _prep(ecg_windows)
        np.savez_compressed(f"{out_dir}/raw_signal.npz", X=outputs["raw_signal"])

    if skna_windows is not None:
        outputs["skna_signal"] = _prep(skna_windows)
        np.savez_compressed(f"{out_dir}/skna_signal.npz", X=outputs["skna_signal"])

    if rskna_windows is not None and iskna_windows is not None:
        rskna_arr = _prep(rskna_windows)
        iskna_arr = _prep(iskna_windows)
        outputs["iskna_rskna"] = {"rSKNA": rskna_arr, "iSKNA": iskna_arr}
        np.savez_compressed(f"{out_dir}/iskna_rskna.npz", rSKNA=rskna_arr, iSKNA=iskna_arr)

    if askna is not None:
        outputs["askna"] = _prep(askna)
        np.savez_compressed(f"{out_dir}/askna.npz", X=outputs["askna"])

    return labels_df, outputs

def calculate_phase_stats(df, time_col='Time_min', value_cols=None):
    df = df.copy()
    cycle_time = df[time_col] % 10
    conditions = [(cycle_time < 4.5), (cycle_time >= 4.5) & (cycle_time < 5.5), (cycle_time >= 5.5)]
    labels = ['Before', 'During', 'After']
    df['Phase'] = np.select(conditions, labels, default='After')
    stats = df.groupby('Phase')[value_cols].agg(['mean', 'std', 'count', 'sem'])
    stats = stats.reindex(labels)
    return stats, df

def fig_statistical_comparison(askna_stats, vitals_stats, fig=None):
    phases = ['Before', 'During', 'After']
    metrics = ['aSKNA', 'SYSTOLIC', 'DIASTOLIC', 'MAP']
    if fig is None:
        fig = Figure(figsize=(10, 7))
    else:
        fig.clear()
    axes = fig.subplots(2, 2)
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        if metric == 'aSKNA':
            stats = askna_stats['aSKNA']
        else:
            matching_col = next((c for c in vitals_stats.columns.levels[0] if metric in c.upper()), None)
            if matching_col is None:
                ax.text(0.5, 0.5, f"{metric} not found", ha='center')
                continue
            stats = vitals_stats[matching_col]

        means = stats.loc[phases, 'mean']
        errors = stats.loc[phases, 'sem']
        ax.bar(phases, means, yerr=errors, capsize=5, color=['#77dd77', '#ffb347', '#779ecb'], alpha=0.8)
        for i, val in enumerate(means):
            error_val = errors.iloc[i]
            ax.text(i, val + error_val + (max(means) * 0.02), f"{val:.2f}", ha='center',
                    fontweight='bold', bbox=dict(facecolor='yellow', alpha=0.3))
        ax.set_title(f"{metric} Comparison")
        ax.set_ylabel("Value")
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    fig.tight_layout()
    return fig

def fig_continuous_segmented_signals(df_askna, df_vitals, fig=None):
    """Draws into `fig` if provided, otherwise creates a new Figure. Returns the Figure used."""
    max_time = df_vitals['Time_min'].max()

    def find_col(keywords):
        for col in df_vitals.columns:
            if any(k.upper() in col.upper() for k in keywords):
                return col
        return None

    cols_to_plot = {
        'aSKNA': ('aSKNA', df_askna['Time_min'], df_askna['aSKNA'], 'r'),
        'SYSTOLIC': (find_col(['SYS', 'SYSTOLIC']), df_vitals['Time_min'], None, 'b'),
        'DIASTOLIC': (find_col(['DIA', 'DIAS', 'DIASTOLIC']), df_vitals['Time_min'], None, 'g'),
        'MAP': (find_col(['MAP']), df_vitals['Time_min'], None, 'purple'),
    }
    for key in ('SYSTOLIC', 'DIASTOLIC', 'MAP'):
        col_name, time, _, color = cols_to_plot[key]
        data = df_vitals[col_name] if col_name else None
        cols_to_plot[key] = (col_name, time, data, color)

    if fig is None:
        fig = Figure(figsize=(10, 9))
    else:
        fig.clear()
    axes = fig.subplots(4, 1, sharex=True)
    for i, (key, (col_name, time, data, color)) in enumerate(cols_to_plot.items()):
        ax = axes[i]
        if col_name is None:
            ax.text(0.5, 0.5, f"{key} not found", ha='center', transform=ax.transAxes)
            continue
        ax.plot(time, data, color=color, linewidth=1.2, label=col_name)
        for cycle_start in range(0, int(max_time) + 1, 10):
            if cycle_start < max_time:
                ax.axvspan(cycle_start, min(cycle_start + 4.5, max_time), color='green', alpha=0.1)
                ax.axvspan(cycle_start + 4.5, min(cycle_start + 5.5, max_time), color='orange', alpha=0.3)
                ax.axvspan(cycle_start + 5.5, min(cycle_start + 10, max_time), color='blue', alpha=0.1)
        ax.set_title(f"{key} Time Series")
        ax.set_ylabel("Value")
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='upper right')
        ax.set_xlim(0, max_time)

    axes[-1].set_xlabel("Time (Minutes)")
    fig.tight_layout()
    return fig