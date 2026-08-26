import sys
import os
import numpy as np
import pandas as pd

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QLineEdit, QFileDialog, QSpinBox, QDoubleSpinBox, QProgressBar,
    QTextEdit, QTabWidget, QTableView, QSplitter, QGroupBox, QFormLayout,
    QCheckBox, QMessageBox, QComboBox, QScrollArea, QGridLayout
)
from PyQt5.QtCore import Qt, QAbstractTableModel
from PyQt5.QtGui import QFont

import pyqtgraph as pg
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import core_processing as core
from worker import PipelineWorker
try:
    from train_worker import LOSOTrainingWorker
    from inference_worker import InferenceWorker
    _TRAIN_WORKER_IMPORT_ERROR = None
except Exception as _e:
    LOSOTrainingWorker = None
    InferenceWorker = None
    _TRAIN_WORKER_IMPORT_ERROR = str(_e)

from scipy.fft import rfft, rfftfreq
from scipy.signal import welch

pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')
pg.setConfigOptions(antialias=False, useOpenGL=False)

class PandasModel(QAbstractTableModel):
    def __init__(self, df=pd.DataFrame()):
        super().__init__()
        self._df = df

    def set_df(self, df):
        self.beginResetModel()
        self._df = df
        self.endResetModel()

    def rowCount(self, parent=None):
        return len(self._df.index)

    def columnCount(self, parent=None):
        return len(self._df.columns)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or role != Qt.DisplayRole:
            return None
        val = self._df.iat[index.row(), index.column()]
        if isinstance(val, float):
            return f"{val:.6g}"
        return str(val)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return str(self._df.columns[section])
        return str(self._df.index[section])


def downsample_for_plot(x, y, max_points=20000):
    """
    Min/max decimation: split into buckets and keep BOTH the min and max
    sample of each bucket (in time order), instead of a plain stride. A
    plain stride can silently skip over the one sample that actually hits
    a signal's true min/max (e.g. the single point that reaches exactly 0
    or 1 after normalize_minmax), making a correctly normalized signal look
    like it never reaches its real extremes.
    """
    n = len(y)
    if n <= max_points:
        return x, y
    bucket_count = max(max_points // 2, 1)
    bucket_size = int(np.ceil(n / bucket_count))
    n_buckets = int(np.ceil(n / bucket_size))
    x_out = np.empty(n_buckets * 2, dtype=float)
    y_out = np.empty(n_buckets * 2, dtype=float)
    for b in range(n_buckets):
        start = b * bucket_size
        end = min(start + bucket_size, n)
        seg_x, seg_y = x[start:end], y[start:end]
        i_min, i_max = int(np.argmin(seg_y)), int(np.argmax(seg_y))
        i_first, i_second = (i_min, i_max) if i_min <= i_max else (i_max, i_min)
        x_out[2 * b], y_out[2 * b] = seg_x[i_first], seg_y[i_first]
        x_out[2 * b + 1], y_out[2 * b + 1] = seg_x[i_second], seg_y[i_second]
    return x_out, y_out


class MplCanvas(FigureCanvas):
    def __init__(self, width=8, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SKNA / Vitals Analysis Pipeline")
        self.resize(1400, 900)

        self.worker = None
        self.results = {}
        self.feature_models = {}

        self.train_worker = None
        self.fold_predictions = {}  # test_subject -> predictions DataFrame, accumulated as folds finish
        self.fold_personalized = {}  # test_subject -> personalized fold_result, accumulated as folds finish
        self.test_worker = None
        self.test_predictions_df = None

        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)

        splitter = QSplitter(Qt.Horizontal)
        root_layout.addWidget(splitter)

        self.shared_control_panel = self._build_control_panel()
        splitter.addWidget(self.shared_control_panel)

        self.tabs = QTabWidget()
        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([340, 1060])

        self._build_signals_tab()
        self._build_windows_tab()
        self._build_features_tab()
        self._build_phase_tab()
        self._build_training_tab()
        self._build_test_tab()

        # The shared Input Files / Parameters / Log panel drives the preprocessing
        # tabs (Signals/Windows/Features/Phase). Model Training and Test Saved
        # Model work off the aggregated dataset and have their own controls, so
        # hide the shared panel there to give them the full window width.
        self.tabs.currentChanged.connect(self._on_main_tab_changed)
        self._on_main_tab_changed(self.tabs.currentIndex())

    def _on_main_tab_changed(self, index):
        hide_on = {"Model Training", "Test Saved Model"}
        self.shared_control_panel.setVisible(self.tabs.tabText(index) not in hide_on)

    def _build_control_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)

        file_group = QGroupBox("Input Files")
        file_form = QFormLayout(file_group)

        self.txt_edit = QLineEdit()
        txt_btn = QPushButton("Browse...")
        txt_btn.clicked.connect(self._browse_txt)
        txt_row = QHBoxLayout()
        txt_row.addWidget(self.txt_edit)
        txt_row.addWidget(txt_btn)
        file_form.addRow("SKNA/ECG .txt:", txt_row)

        self.csv_edit = QLineEdit()
        csv_btn = QPushButton("Browse...")
        csv_btn.clicked.connect(self._browse_csv)
        csv_row = QHBoxLayout()
        csv_row.addWidget(self.csv_edit)
        csv_row.addWidget(csv_btn)
        file_form.addRow("Vitals .csv:", csv_row)

        self.subject_id_edit = QLineEdit()
        self.subject_id_edit.setPlaceholderText("e.g. S01")
        file_form.addRow("Subject ID:", self.subject_id_edit)

        self.session_id_edit = QLineEdit()
        self.session_id_edit.setPlaceholderText("e.g. session1 (optional)")
        file_form.addRow("Session ID:", self.session_id_edit)

        self.out_edit = QLineEdit(os.path.abspath("feature_result"))
        out_btn = QPushButton("Browse...")
        out_btn.clicked.connect(self._browse_out)
        out_row = QHBoxLayout()
        out_row.addWidget(self.out_edit)
        out_row.addWidget(out_btn)
        file_form.addRow("Output dir:", out_row)

        out_hint = QLabel("Creates '<subject>_<session>/windowing_labeling/' and\n'<subject>_<session>/features/' subfolders inside it.")
        out_hint.setStyleSheet("color: gray; font-size: 10px;")
        file_form.addRow("", out_hint)

        layout.addWidget(file_group)

        param_group = QGroupBox("Parameters")
        param_form = QFormLayout(param_group)

        self.fs_spin = QSpinBox()
        self.fs_spin.setRange(1, 100000)
        self.fs_spin.setValue(10000)
        param_form.addRow("Sampling rate (Hz):", self.fs_spin)

        self.window_sec_spin = QSpinBox()
        self.window_sec_spin.setRange(1, 3600)
        self.window_sec_spin.setValue(5)
        param_form.addRow("Window length (s):", self.window_sec_spin)

        # --- high-pass filter of the raw signal (very first processing step) ---
        # Butterworth HPF (default 0.08 Hz / order 2) to remove baseline wander
        # from the raw signal - mainly the ECG that becomes the raw_signal dataset.
        self.hpf_check = QCheckBox("High-pass filter raw signal (Butterworth)")
        self.hpf_check.setChecked(True)
        self.hpf_check.setToolTip(
            "Remove slow baseline wander / DC drift from the raw signal before "
            "anything else. Mainly affects the ECG (raw_signal dataset) and the "
            "raw-signal plots; the SKNA 500-1000 Hz band-pass masks it there.")
        param_form.addRow(self.hpf_check)

        self.hpf_fc_spin = QDoubleSpinBox()
        self.hpf_fc_spin.setRange(0.001, 100.0)
        self.hpf_fc_spin.setDecimals(3)
        self.hpf_fc_spin.setValue(0.08)
        self.hpf_fc_spin.setSuffix(" Hz")
        self.hpf_fc_spin.setToolTip("High-pass cutoff frequency.")
        param_form.addRow("HPF cutoff:", self.hpf_fc_spin)

        self.hpf_order_spin = QSpinBox()
        self.hpf_order_spin.setRange(1, 8)
        self.hpf_order_spin.setValue(2)
        self.hpf_order_spin.setToolTip("Butterworth filter order.")
        param_form.addRow("HPF order:", self.hpf_order_spin)

        self.detrend_full_check = QCheckBox("Detrend full CH1/CH40/CH41 before windowing")
        self.detrend_full_check.setChecked(True)
        param_form.addRow(self.detrend_full_check)

        self.detrend_full_type_combo = QComboBox()
        self.detrend_full_type_combo.addItems(["linear", "constant"])
        param_form.addRow("Full-signal detrend type:", self.detrend_full_type_combo)

        self.detrend_window_check = QCheckBox("Also detrend per-window in exported CSVs")
        self.detrend_window_check.setChecked(False)
        param_form.addRow(self.detrend_window_check)

        self.detrend_window_type_combo = QComboBox()
        self.detrend_window_type_combo.addItems(["linear", "constant"])
        param_form.addRow("Per-window detrend type:", self.detrend_window_type_combo)

        self.npz_output_check = QCheckBox("Save labeled datasets as compact .npz (recommended)")
        self.npz_output_check.setChecked(True)
        self.npz_output_check.setToolTip(
            "Off = legacy wide CSV (one column per sample, large files).\n"
            "On = labels.csv + compressed .npz arrays (float32), typically 5-10x smaller.")
        param_form.addRow(self.npz_output_check)

        self.cwt_check = QCheckBox("Compute CWT features")
        self.cwt_check.setChecked(False)
        param_form.addRow(self.cwt_check)

        self.cwt_save_images_check = QCheckBox("Save CWT scalogram images (slow)")
        self.cwt_save_images_check.setChecked(False)
        param_form.addRow(self.cwt_save_images_check)

        self.cwt_window_spin = QSpinBox()
        self.cwt_window_spin.setRange(3, 7200)
        self.cwt_window_spin.setValue(600)
        param_form.addRow("CWT window length (s):", self.cwt_window_spin)

        self.cwt_arrays_check = QCheckBox("Compute CWT arrays for training (5s windows)")
        self.cwt_arrays_check.setChecked(False)
        self.cwt_arrays_check.setToolTip(
            "Computes a CWT scalogram for every 5s window (same windows used for "
            "SKNA/rSKNA/iSKNA training), log-compressed + normalized + average-pooled "
            "along time down to 'CWT time bins' columns, quantized to uint8 grayscale, "
            "and saved as one compressed cwt_signal.npz (matching skna_signal.npz etc.) "
            "in the labels/windowing output folder - a DL-ready input, separate from the "
            "exploratory 'Compute CWT features' option above (which uses a much longer, "
            "non-training window).")
        param_form.addRow(self.cwt_arrays_check)

        self.cwt_time_bins_spin = QSpinBox()
        self.cwt_time_bins_spin.setRange(8, 2048)
        self.cwt_time_bins_spin.setValue(128)
        self.cwt_time_bins_spin.setToolTip(
            "Number of time columns each 5s window's CWT array is pooled down to. "
            "128 is a good default - well above typical SKNA burst duration, but a "
            "~390x storage reduction vs. full resolution (50000 columns at fs=10000Hz).")
        param_form.addRow("CWT time bins (per 5s window):", self.cwt_time_bins_spin)

        self.fd_arrays_check = QCheckBox("Compute FD arrays for training (5s windows)")
        self.fd_arrays_check.setChecked(False)
        self.fd_arrays_check.setToolTip(
            "Computes a frequency-domain array for every 5s window: FFT magnitude and "
            "Welch PSD, each sliced to the 500-999 Hz passband and log-normalized, stacked "
            "as a 2-channel (FFT, PSD) vector, and saved as one compressed fd_signal.npz "
            "(matching skna_signal.npz etc.) in the labels/windowing output folder - the "
            "input for the model's optional frequency-domain 1D CNN branch.")
        param_form.addRow(self.fd_arrays_check)

        layout.addWidget(param_group)

        run_row = QHBoxLayout()
        self.run_btn = QPushButton("Run Pipeline")
        self.run_btn.clicked.connect(self._run_pipeline)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self._cancel_pipeline)
        self.cancel_btn.setEnabled(False)
        run_row.addWidget(self.run_btn)
        run_row.addWidget(self.cancel_btn)
        layout.addLayout(run_row)

        self.progress_bar = QProgressBar()
        self.progress_label = QLabel("Idle")
        layout.addWidget(self.progress_label)
        layout.addWidget(self.progress_bar)

        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout(log_group)
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setFont(QFont("Monospace", 9))
        log_layout.addWidget(self.log_edit)
        layout.addWidget(log_group, stretch=1)

        return panel

    def _build_signals_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.signal_plots = pg.GraphicsLayoutWidget()
        layout.addWidget(self.signal_plots)

        self.plot_raw = self.signal_plots.addPlot(row=0, col=0, title="Raw vs detrended channels (CH1 / CH40 / CH41)")
        self.plot_raw.addLegend()
        self.plot_raw.setLabel('bottom', 'Time', 'min')
        self.signal_plots.nextRow()

        self.plot_filtered = self.signal_plots.addPlot(row=1, col=0, title="Filtered SKNA (500-1000 Hz)")
        self.plot_filtered.setLabel('bottom', 'Time', 'min')
        self.signal_plots.nextRow()

        self.plot_rskna = self.signal_plots.addPlot(row=2, col=0, title="rSKNA (rectified)")
        self.plot_rskna.setLabel('bottom', 'Time', 'min')
        self.signal_plots.nextRow()

        self.plot_askna = self.signal_plots.addPlot(row=3, col=0, title="aSKNA (window average)")
        self.plot_askna.setLabel('bottom', 'Time', 'min')
        self.signal_plots.nextRow()

        self.plot_vitals = self.signal_plots.addPlot(row=4, col=0, title="Vitals (SYS / DIA / MAP)")
        self.plot_vitals.addLegend()
        self.plot_vitals.setLabel('bottom', 'Time', 'min')

        for p in (self.plot_filtered, self.plot_rskna, self.plot_askna, self.plot_vitals):
            p.setXLink(self.plot_raw)

        self.tabs.addTab(tab, "Signals")

    def _build_windows_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        ctrl_row = QHBoxLayout()
        ctrl_row.addWidget(QLabel("Window index:"))
        self.window_spin = QSpinBox()
        self.window_spin.setMinimum(0)
        self.window_spin.valueChanged.connect(self._update_window_view)
        ctrl_row.addWidget(self.window_spin)
        self.window_count_label = QLabel("of 0")
        ctrl_row.addWidget(self.window_count_label)
        ctrl_row.addStretch(1)
        layout.addLayout(ctrl_row)

        splitter = QSplitter(Qt.Vertical)
        layout.addWidget(splitter)

        self.win_time_plots = pg.GraphicsLayoutWidget()
        self.win_plot_skna = self.win_time_plots.addPlot(row=0, col=0, title="Filtered SKNA")
        self.win_plot_rskna = self.win_time_plots.addPlot(row=0, col=1, title="rSKNA")
        self.win_plot_iskna = self.win_time_plots.addPlot(row=0, col=2, title="iSKNA")
        splitter.addWidget(self.win_time_plots)

        self.win_freq_plots = pg.GraphicsLayoutWidget()
        self.win_plot_fft = self.win_freq_plots.addPlot(row=0, col=0, title="FFT magnitude")
        self.win_plot_psd = self.win_freq_plots.addPlot(row=0, col=1, title="PSD (Welch)", )
        # Linear y-axis: PSD magnitude is normalized to [0, 1] per window (see
        # core.normalize_minmax), and log-mode would break on the window's
        # minimum, which normalizes to exactly 0.
        splitter.addWidget(self.win_freq_plots)

        cwt_row = QWidget()
        cwt_layout = QVBoxLayout(cwt_row)
        cwt_btn_row = QHBoxLayout()
        self.cwt_preview_btn = QPushButton("Compute CWT scalogram for this window")
        self.cwt_preview_btn.clicked.connect(self._compute_cwt_preview)
        cwt_btn_row.addWidget(self.cwt_preview_btn)
        cwt_btn_row.addStretch(1)
        cwt_layout.addLayout(cwt_btn_row)
        self.cwt_canvas = MplCanvas(width=8, height=3)
        cwt_layout.addWidget(self.cwt_canvas)
        splitter.addWidget(cwt_row)

        self.tabs.addTab(tab, "Windows")

    def _build_features_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        top_row = QHBoxLayout()
        top_row.addWidget(QLabel("Feature set:"))
        self.feature_combo = QComboBox()
        self.feature_combo.currentTextChanged.connect(self._show_feature_table)
        top_row.addWidget(self.feature_combo)
        top_row.addStretch(1)
        export_btn = QPushButton("Export current table to CSV...")
        export_btn.clicked.connect(self._export_current_feature)
        top_row.addWidget(export_btn)
        layout.addLayout(top_row)

        self.feature_table = QTableView()
        layout.addWidget(self.feature_table)

        self.tabs.addTab(tab, "Features")

    def _build_phase_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        splitter = QSplitter(Qt.Vertical)
        layout.addWidget(splitter)

        self.phase_bar_canvas = MplCanvas(width=8, height=5)
        splitter.addWidget(self.phase_bar_canvas)

        self.phase_series_canvas = MplCanvas(width=8, height=6)
        splitter.addWidget(self.phase_series_canvas)

        btn_row = QHBoxLayout()
        save_btn = QPushButton("Save both figures...")
        save_btn.clicked.connect(self._save_phase_figures)
        btn_row.addWidget(save_btn)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        self.tabs.addTab(tab, "Phase Comparison")

    def _build_training_tab(self):
        tab = QWidget()
        outer = QHBoxLayout(tab)

        if _TRAIN_WORKER_IMPORT_ERROR is not None:
            err_label = QLabel(
                "Could not load the training module (PyTorch failed to import):\n\n"
                f"{_TRAIN_WORKER_IMPORT_ERROR}\n\n"
                "This is usually a broken/incompatible PyTorch install on this machine "
                "(e.g. missing Visual C++ Redistributable on Windows, or a CUDA build "
                "without a matching driver). The rest of the app is unaffected.\n\n"
                "Fix the PyTorch install, then restart the app."
            )
            err_label.setWordWrap(True)
            err_label.setStyleSheet("color: #b00020; padding: 20px;")
            outer.addWidget(err_label)
            self.tabs.addTab(tab, "Model Training")
            return

        # --- Left: controls (self-contained, since training operates on the
        # aggregated multi-subject output, not the single txt/csv used above) ---
        control_panel = QWidget()
        cl = QVBoxLayout(control_panel)

        data_group = QGroupBox("Aggregated dataset")
        data_form = QFormLayout(data_group)
        self.combined_dir_edit = QLineEdit()
        # Small minimum so the field yields space to the Browse button instead of
        # pushing it off the (narrow, scrollable) panel's right edge.
        self.combined_dir_edit.setMinimumWidth(80)
        combined_btn = QPushButton("Browse...")
        combined_btn.clicked.connect(self._browse_combined_dir)
        combined_row = QHBoxLayout()
        combined_row.addWidget(self.combined_dir_edit)
        combined_row.addWidget(combined_btn)
        data_form.addRow("Combined dir:", combined_row)
        combined_hint = QLabel("From aggregate_datasets.py --format npz\n"
                                "(expects all_raw_signal.npz + labels, etc.)")
        combined_hint.setStyleSheet("color: gray; font-size: 10px;")
        data_form.addRow("", combined_hint)

        self.train_out_edit = QLineEdit(os.path.abspath("training_results"))
        self.train_out_edit.setMinimumWidth(80)
        train_out_btn = QPushButton("Browse...")
        train_out_btn.clicked.connect(self._browse_train_out)
        train_out_row = QHBoxLayout()
        train_out_row.addWidget(self.train_out_edit)
        train_out_row.addWidget(train_out_btn)
        data_form.addRow("Checkpoints dir:", train_out_row)
        cl.addWidget(data_group)

        seq_group = QGroupBox("Sequence")
        seq_form = QFormLayout(seq_group)
        self.seq_len_spin = QSpinBox()
        self.seq_len_spin.setRange(2, 500)
        self.seq_len_spin.setValue(10)
        seq_form.addRow("Sequence length (windows):", self.seq_len_spin)
        self.stride_spin = QSpinBox()
        self.stride_spin.setRange(1, 500)
        self.stride_spin.setValue(1)
        seq_form.addRow("Stride:", self.stride_spin)
        self.downsample_spin = QSpinBox()
        self.downsample_spin.setRange(1, 50)
        self.downsample_spin.setValue(5)
        self.downsample_spin.setToolTip(
            "Integer factor by which the 1D time-domain signals (ECG/SKNA/iSKNA/rSKNA) are "
            "decimated ONCE at load time - you do NOT need to pre-downsample each recording's "
            "file. An anti-aliased zero-phase FIR low-pass runs before subsampling, so there is "
            "no aliasing; content above the new Nyquist ((fs/2)/factor) is cleanly removed.\n\n"
            "At fs=10 kHz, 5 s windows (50000 samples):\n"
            "  1  = off (full 10 kHz / 50000 samples) - highest memory, can OOM-crash.\n"
            "  2  -> 5 kHz  / 25000 samples (keeps 0-2500 Hz; safest for sKNA).\n"
            "  5  -> 2 kHz  / 10000 samples (keeps 0-1000 Hz, the classic sKNA band; ~5x less "
            "RAM & CNN memory - recommended).\n"
            " 10 -> 1 kHz  /  5000 samples (drops the 500-1000 Hz sKNA band; ECG-only runs only).\n\n"
            "aSKNA (one value/window) and FD/CWT (spectral features) are NOT decimated. The "
            "factor is saved in the checkpoint, so inference decimates identically.")
        seq_form.addRow("Downsample factor (÷):", self.downsample_spin)

        self.val_mode_combo = QComboBox()
        self.val_mode_combo.addItem("Subject holdout (honest, nested LOSO)", "subject_holdout")
        self.val_mode_combo.addItem("Time-split (fast, optimistic)", "time_split")
        self.val_mode_combo.setCurrentIndex(0)
        self.val_mode_combo.setToolTip(
            "How the validation set used for early stopping/model selection is chosen, per fold:\n\n"
            "Subject holdout: picks ONE whole subject out of the remaining (non-test) pool "
            "and excludes it entirely from training, using it only for validation - a nested "
            "LOSO. Noisier early stopping, but val_MAE genuinely previews how the model does "
            "on someone it never trained on, same as the held-out TEST subject.\n\n"
            "Time-split: pulls the LAST 'validation fraction' of EVERY remaining subject's own "
            "recording (by time). Fast/stable, but the model already trained on the earlier "
            "part of that same session, so it's measuring within-session interpolation, not "
            "generalization to a new person - val_MAE tends to look much better than the real "
            "held-out TEST subject's score.")
        self.val_mode_combo.currentIndexChanged.connect(self._on_val_mode_changed)
        seq_form.addRow("Validation strategy:", self.val_mode_combo)

        self.val_frac_spin = QDoubleSpinBox()
        self.val_frac_spin.setDecimals(2)
        self.val_frac_spin.setRange(0.05, 0.5)
        self.val_frac_spin.setSingleStep(0.05)
        self.val_frac_spin.setValue(0.15)
        self.val_frac_spin.setToolTip(
            "Only used by the 'Time-split' validation strategy. Fraction of EACH "
            "remaining subject's recording (by time) held out for validation/"
            "early-stopping, per fold.")
        seq_form.addRow("Validation fraction (time-split only):", self.val_frac_spin)

        self.norm_mode_combo = QComboBox()
        self.norm_mode_combo.addItem("Global (one mean/std for all subjects)", "global")
        self.norm_mode_combo.addItem("Per-recording calibration windows", "calib")
        self.norm_mode_combo.setCurrentIndex(0)
        self.norm_mode_combo.setToolTip(
            "How the ECG/SKNA inputs are z-scored before the CNN branches:\n\n"
            "Global: one mean/std per channel, fit on the TRAIN subjects and reused for "
            "val/test. A held-out subject whose signal sits at a different absolute scale "
            "gets read through another population's statistics.\n\n"
            "Per-recording calibration: each recording is z-scored by the mean/std of its "
            "OWN first 'calibration windows' - the input-side mirror of the calibration-"
            "relative BP target. Uses only that recording's own early samples, so it is not "
            "LOSO leakage, for the same reason the BP baseline isn't. Try this when the "
            "model tracks BP well within a subject but collapses toward the mean on a "
            "held-out one.")
        seq_form.addRow("Input normalization:", self.norm_mode_combo)

        self.exclude_subjects_edit = QLineEdit()
        self.exclude_subjects_edit.setPlaceholderText("e.g. s5  (comma-separated, blank = keep all)")
        self.exclude_subjects_edit.setToolTip(
            "Subjects dropped from the entire LOSO run - they never appear as train, "
            "validation, or test.\n\n"
            "Because the model's fallback output is the training-set mean "
            "delta-from-calibration, one subject with an atypical mean delta skews EVERY "
            "fold, not just its own. Use this to ablate such a subject, or to quarantine "
            "a recording whose BP labels are suspect.")
        seq_form.addRow("Exclude subjects:", self.exclude_subjects_edit)

        cl.addWidget(seq_group)
        self._on_val_mode_changed()

        hp_group = QGroupBox("Training")
        hp_form = QFormLayout(hp_group)
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 1024)
        self.batch_size_spin.setValue(16)
        hp_form.addRow("Batch size:", self.batch_size_spin)

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 5000)
        self.epochs_spin.setValue(50)
        hp_form.addRow("Max epochs:", self.epochs_spin)

        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(1, 500)
        self.patience_spin.setValue(10)
        hp_form.addRow("Early stop patience:", self.patience_spin)

        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setDecimals(5)
        self.lr_spin.setRange(0.00001, 1.0)
        self.lr_spin.setSingleStep(0.0001)
        self.lr_spin.setValue(0.001)
        hp_form.addRow("Learning rate:", self.lr_spin)

        self.weight_decay_spin = QDoubleSpinBox()
        self.weight_decay_spin.setDecimals(5)
        self.weight_decay_spin.setRange(0.0, 1.0)
        self.weight_decay_spin.setSingleStep(0.0001)
        self.weight_decay_spin.setValue(0.0001)
        self.weight_decay_spin.setToolTip(
            "L2 regularization on the optimizer. Higher = stronger penalty on large "
            "weights, helps fight overfitting with few subjects. Try 1e-3 to 1e-2 "
            "if predictions cluster into a few discrete values instead of tracking true BP.")
        hp_form.addRow("Weight decay:", self.weight_decay_spin)

        self.lstm_hidden_spin = QSpinBox()
        self.lstm_hidden_spin.setRange(4, 1024)
        self.lstm_hidden_spin.setValue(64)
        hp_form.addRow("BiLSTM hidden size:", self.lstm_hidden_spin)

        self.cnn_layers_spin = QSpinBox()
        self.cnn_layers_spin.setRange(1, 4)
        self.cnn_layers_spin.setValue(3)
        self.cnn_layers_spin.setToolTip(
            "Number of Conv1d->BN->ReLU->MaxPool blocks stacked in EACH CNN branch "
            "(depth, not the number of branches). Channels double per block starting "
            "from 16 (e.g. 2 layers = 16,32; 3 layers = 16,32,64). More depth = more "
            "capacity but overfits faster with few LOSO subjects. Original model used 2.")
        hp_form.addRow("CNN conv layers / branch:", self.cnn_layers_spin)

        self.cnn_arch_combo = QComboBox()
        self.cnn_arch_combo.addItem("Plain CNN (original)", "plain")
        self.cnn_arch_combo.addItem("1D ResNet (residual)", "resnet")
        self.cnn_arch_combo.addItem("ResNet-18 1D (deep, 64->512)", "resnet18")
        self.cnn_arch_combo.addItem("ANN / MLP (no convolution)", "ann")
        self.cnn_arch_combo.setCurrentIndex(0)
        self.cnn_arch_combo.setToolTip(
            "Architecture of the 1D branches: the raw waveforms (ECG / SKNA / iSKNA+rSKNA). The "
            "2D CWT scalogram branch is chosen separately below. FD / FFT / PSD (coarse spectra) "
            "stay plain CNNs under every convolutional option here.\n\n"
            "Plain CNN: the original stacked Conv1d->BN->ReLU->MaxPool blocks.\n\n"
            "1D ResNet: residual blocks with a strided stem (one stage per 'CNN conv layer', same "
            "channel widths, same output size). Residual skips keep depth trainable; memory stays "
            "comparable (a little higher than Plain, and bounded by gradient checkpointing either "
            "way). With few LOSO subjects, extra capacity overfits, so compare it against Plain on "
            "the same folds before going bigger.\n\n"
            "ResNet-18 1D: the canonical ResNet-18 topology - 4 stages of [2,2,2,2] residual "
            "blocks at the standard widths (64,128,256,512) per waveform branch. Those fixed "
            "widths are part of the ResNet-18 definition, so this ignores the per-branch channel "
            "settings. Much larger than the other two - with few LOSO subjects it will overfit "
            "unless strongly regularized; judge it only by held-out (LOSO) MAE, never train "
            "loss.\n\n"
            "ANN / MLP: no convolution at all - each window is adaptive-average-pooled down to "
            "'ANN pool length' samples, flattened, and run through Linear->BN->ReLU->Dropout "
            "layers whose widths are the per-branch channel settings. The pooling is what keeps "
            "the first Linear small: a raw window can be tens of thousands of samples, and "
            "flattening that directly would be a multi-million-parameter layer per branch. This "
            "is the shallow fully-connected baseline you compare the conv models against, so "
            "unlike the conv options it also replaces the FD / FFT / PSD branches with MLPs.\n\n"
            "The choice is saved in the checkpoint, so inference rebuilds the matching model.")
        hp_form.addRow("1D branch arch:", self.cnn_arch_combo)

        self.cwt_arch_combo = QComboBox()
        self.cwt_arch_combo.addItem("Plain 2D CNN (original)", "plain")
        self.cwt_arch_combo.addItem("2D ResNet (residual)", "resnet")
        self.cwt_arch_combo.addItem("ResNet-18 2D (deep, 64->512)", "resnet18")
        self.cwt_arch_combo.setCurrentIndex(0)
        self.cwt_arch_combo.setToolTip(
            "Architecture of the 2D branch - the CWT scalogram. Only used when the CWT input is "
            "enabled, and set independently of the 1D branch arch, so combinations like "
            "ANN on the waveforms + ResNet-18 2D on the scalogram are available.\n\n"
            "Plain 2D CNN: the original stacked Conv2d->BN->ReLU->MaxPool blocks at the CWT "
            "channel widths.\n\n"
            "2D ResNet: residual blocks with a strided stem, one stage per CWT channel width. The "
            "scalogram IS a 2D image, so image-style residual nets genuinely fit here.\n\n"
            "ResNet-18 2D: the canonical ResNet-18 - 4 stages of [2,2,2,2] BasicBlocks at widths "
            "(64,128,256,512) over the scalogram. Those widths are part of the definition, so "
            "this ignores the CWT channel settings. Same overfitting caveat as the 1D version: "
            "judge it only by held-out (LOSO) MAE.")
        hp_form.addRow("2D (CWT) branch arch:", self.cwt_arch_combo)

        self.ann_pool_len_spin = QSpinBox()
        self.ann_pool_len_spin.setRange(8, 4096)
        self.ann_pool_len_spin.setValue(256)
        self.ann_pool_len_spin.setToolTip(
            "ANN / MLP branches only (ignored by every convolutional arch). Each window is "
            "adaptive-average-pooled to this many samples before being flattened into the MLP, so "
            "the first Linear has in_channels * this many inputs regardless of window length. "
            "Larger = finer temporal detail reaches the MLP but a bigger first layer; smaller = "
            "coarser and more heavily regularized by the pooling itself.")
        hp_form.addRow("ANN pool length:", self.ann_pool_len_spin)

        self.lstm_layers_spin = QSpinBox()
        self.lstm_layers_spin.setRange(1, 4)
        self.lstm_layers_spin.setValue(2)
        self.lstm_layers_spin.setToolTip(
            "Number of stacked BiLSTM layers. Original model used 1. Dropout is applied "
            "between layers only when this is > 1.")
        hp_form.addRow("BiLSTM layers:", self.lstm_layers_spin)

        self.dropout_spin = QDoubleSpinBox()
        self.dropout_spin.setDecimals(2)
        self.dropout_spin.setRange(0.0, 0.9)
        self.dropout_spin.setSingleStep(0.05)
        self.dropout_spin.setValue(0.3)
        hp_form.addRow("Dropout:", self.dropout_spin)

        self.anti_collapse_lambda_spin = QDoubleSpinBox()
        self.anti_collapse_lambda_spin.setDecimals(2)
        self.anti_collapse_lambda_spin.setRange(0.0, 5.0)
        self.anti_collapse_lambda_spin.setSingleStep(0.1)
        self.anti_collapse_lambda_spin.setValue(0.0)
        self.anti_collapse_lambda_spin.setToolTip(
            "Weight on the anti-collapse penalty (loss = Huber + lambda * "
            "relu(target_std - pred_std) per batch). Directly punishes the model "
            "for varying its output less than the true BP does - the mean-collapse "
            "failure evaluate() flags as 'model barely varies its output'. "
            "0 = pure Huber loss, no penalty. Off by default: this penalty can "
            "inflate output variance without it correlating to the target, which "
            "masks a non-tracking model rather than fixing it - only enable it "
            "once you've confirmed the model has real tracking skill (see "
            "MAE_debiased / MAE_oracle_const in the LOSO summary) and just needs "
            "wider output spread.")
        hp_form.addRow("Anti-collapse lambda:", self.anti_collapse_lambda_spin)

        self.use_gpu_check = QCheckBox("Use GPU if available")
        self.use_gpu_check.setChecked(True)
        hp_form.addRow(self.use_gpu_check)

        self.personalize_check = QCheckBox("Personalize (per-subject fine-tune)")
        self.personalize_check.setChecked(False)
        self.personalize_check.setToolTip(
            "After training each fold's population (LOSO) model, additionally fine-tune it "
            "on the held-out subject's OWN early data (hybrid calibration: freeze all but the "
            "last LSTM layer + attention + head). Each subject's recording is split by time into "
            "adapt (<50%), val (50-60%), and a held-out test (>60%) slice; the report adds "
            "p_base_* vs p_post_* MAE and p_corr_*_delta on that test slice. NOTE: these "
            "personalized numbers are on the smaller held-out slice, so compare them to "
            "p_base_* (same slice), not to the plain-LOSO MAE columns. Default off.")
        hp_form.addRow(self.personalize_check)

        # Fine-tune (personalization) knobs. These mirror the CLI flags of the
        # standalone personalize_finetune.py so a GUI run is just as tunable.
        # They stay disabled (and fall back to the worker's defaults) unless
        # Personalize is on. adapt_end/val_end are the time-split boundaries of
        # each held-out subject's recording: adapt<adapt_end, val in
        # [adapt_end, val_end), test>=val_end.
        self.adapt_end_spin = QDoubleSpinBox()
        self.adapt_end_spin.setDecimals(2)
        self.adapt_end_spin.setRange(0.10, 0.90)
        self.adapt_end_spin.setSingleStep(0.05)
        self.adapt_end_spin.setValue(0.50)
        self.adapt_end_spin.setToolTip(
            "End of the adapt (calibration) slice as a fraction of each held-out "
            "subject's recording. Windows before this feed the per-subject "
            "fine-tune. Must be < val split end.")
        hp_form.addRow("FT adapt split end:", self.adapt_end_spin)

        self.val_end_spin = QDoubleSpinBox()
        self.val_end_spin.setDecimals(2)
        self.val_end_spin.setRange(0.15, 0.95)
        self.val_end_spin.setSingleStep(0.05)
        self.val_end_spin.setValue(0.60)
        self.val_end_spin.setToolTip(
            "End of the fine-tune validation slice (early-stopping signal), as a "
            "fraction of the recording. Windows in [adapt end, val end) are val; "
            "everything from val end on is the held-out personalization test "
            "slice. Must be > adapt split end.")
        hp_form.addRow("FT val split end:", self.val_end_spin)

        self.ft_lr_spin = QDoubleSpinBox()
        self.ft_lr_spin.setDecimals(5)
        self.ft_lr_spin.setRange(0.00001, 1.0)
        self.ft_lr_spin.setSingleStep(0.0001)
        self.ft_lr_spin.setValue(0.0002)
        self.ft_lr_spin.setToolTip(
            "Learning rate for the per-subject fine-tune. Lower than the base LR "
            "since only the last LSTM layer + attention + head are unfrozen and "
            "the adapt slice is small.")
        hp_form.addRow("FT learning rate:", self.ft_lr_spin)

        self.ft_epochs_spin = QSpinBox()
        self.ft_epochs_spin.setRange(1, 500)
        self.ft_epochs_spin.setValue(40)
        self.ft_epochs_spin.setToolTip("Max epochs for the per-subject fine-tune (early-stopped on the FT val slice).")
        hp_form.addRow("FT max epochs:", self.ft_epochs_spin)

        self.ft_patience_spin = QSpinBox()
        self.ft_patience_spin.setRange(1, 200)
        self.ft_patience_spin.setValue(8)
        self.ft_patience_spin.setToolTip("Early-stop patience (in FT epochs) for the per-subject fine-tune.")
        hp_form.addRow("FT patience:", self.ft_patience_spin)

        self._ft_knob_spins = [self.adapt_end_spin, self.val_end_spin,
                               self.ft_lr_spin, self.ft_epochs_spin, self.ft_patience_spin]
        self.personalize_check.toggled.connect(self._on_personalize_toggled)
        self._on_personalize_toggled(self.personalize_check.isChecked())

        cl.addWidget(hp_group)

        # --- CNN inputs (own group so the training knobs above stay readable) ---
        # Preset dropdown auto-fills the per-branch checkboxes; hand-editing
        # any box flips the preset to "Custom". See _apply_input_preset /
        # _mark_input_preset_custom. Labels are short (details live in tooltips)
        # so they all fit a compact 2-column grid.
        input_group = QGroupBox("CNN inputs")
        input_layout = QVBoxLayout(input_group)
        self._applying_preset = False

        self.input_preset_combo = QComboBox()
        for label, key in [("All signals", "all"), ("ECG only", "ecg_only"),
                           ("SKNA only (all SKNA)", "skna_only"),
                           ("ECG + filtered SKNA", "ecg_skna"), ("Custom", "custom")]:
            self.input_preset_combo.addItem(label, key)
        self.input_preset_combo.setToolTip(
            "Which signals feed the CNN. Presets tick the boxes below; editing a "
            "box switches this to 'Custom'. Use this to ablate inputs, e.g. train "
            "on ECG only or SKNA only and compare MAE / ME +/- SD.")
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Input set:"))
        preset_row.addWidget(self.input_preset_combo, 1)
        input_layout.addLayout(preset_row)

        self.use_ecg_check = QCheckBox("ECG")
        self.use_skna_check = QCheckBox("Filtered SKNA")
        self.use_iskna_rskna_check = QCheckBox("iSKNA + rSKNA")
        self.use_askna_check = QCheckBox("aSKNA")
        self.use_fft_check = QCheckBox("FFT")
        self.use_psd_check = QCheckBox("PSD")
        self.use_cwt_check = QCheckBox("CWT (scalogram)")
        for cb in (self.use_ecg_check, self.use_skna_check,
                   self.use_iskna_rskna_check, self.use_askna_check):
            cb.setChecked(True)
        self.use_askna_check.setToolTip(
            "Append the 3s-average aSKNA scalar as an extra input feature.")
        _fd_common_tip = (
            "Requires all_fd_signal.npz in the aggregated dataset (enable 'Compute FD arrays' "
            "in preprocessing, then re-aggregate). FFT and PSD are transforms of the same SKNA "
            "window, so these add an easier-to-learn representation, not new information - with "
            "few LOSO subjects they may not help; compare against TD-only. FFT and PSD are now "
            "independent branches; tick either, both, or neither.")
        self.use_fft_check.setToolTip(
            "Adds a 1D CNN branch over each window's FFT-magnitude passband spectrum "
            "(channel 0 of the FD array).\n\n" + _fd_common_tip)
        self.use_psd_check.setToolTip(
            "Adds a 1D CNN branch over each window's Welch-PSD passband spectrum "
            "(channel 1 of the FD array).\n\n" + _fd_common_tip)
        self.use_cwt_check.setToolTip(
            "Adds a 2D CNN branch over each window's CWT scalogram (a time-frequency image). "
            "Requires all_cwt_signal.npz in the aggregated dataset (enable 'Compute CWT arrays' "
            "in preprocessing, then re-aggregate). Same caveat as the FD branch: extra capacity "
            "that can overfit with few subjects - A/B it against TD-only.")

        # Core time-domain signals in the left column, spectral/extra branches right.
        input_grid = QGridLayout()
        input_grid.setHorizontalSpacing(16)
        input_grid.addWidget(self.use_ecg_check,         0, 0)
        input_grid.addWidget(self.use_askna_check,       0, 1)
        input_grid.addWidget(self.use_skna_check,        1, 0)
        input_grid.addWidget(self.use_fft_check,         1, 1)
        input_grid.addWidget(self.use_iskna_rskna_check, 2, 0)
        input_grid.addWidget(self.use_psd_check,         2, 1)
        input_grid.addWidget(self.use_cwt_check,         3, 1)
        input_grid.setColumnStretch(0, 1)
        input_grid.setColumnStretch(1, 1)
        input_layout.addLayout(input_grid)
        cl.addWidget(input_group)

        # Two-way sync between the preset dropdown and the six input checkboxes.
        self._input_checks = [
            self.use_ecg_check, self.use_skna_check, self.use_iskna_rskna_check,
            self.use_askna_check, self.use_fft_check, self.use_psd_check,
            self.use_cwt_check]
        self.input_preset_combo.currentIndexChanged.connect(self._apply_input_preset)
        for cb in self._input_checks:
            cb.toggled.connect(self._mark_input_preset_custom)

        # The 2D branch arch only bites when the CWT input is on, and the ANN pool
        # length only when the 1D arch is the ANN - grey each out otherwise so the
        # setting can't look as though it's doing something it isn't.
        self.use_cwt_check.toggled.connect(self.cwt_arch_combo.setEnabled)
        self.cwt_arch_combo.setEnabled(self.use_cwt_check.isChecked())
        self.cnn_arch_combo.currentIndexChanged.connect(
            lambda _: self.ann_pool_len_spin.setEnabled(
                self.cnn_arch_combo.currentData() == "ann"))
        self.ann_pool_len_spin.setEnabled(self.cnn_arch_combo.currentData() == "ann")

        cl.addStretch(1)

        # Scroll the config groups so a short window never clips them, and pin
        # Run/Cancel + progress below the scroll area so they stay reachable.
        config_scroll = QScrollArea()
        config_scroll.setWidgetResizable(True)
        config_scroll.setWidget(control_panel)
        # AsNeeded (not AlwaysOff): if a row is ever wider than the panel, show a
        # scrollbar instead of silently clipping controls off the right edge.
        config_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        left_col = QWidget()
        left_col.setMinimumWidth(300)
        left_col.setMaximumWidth(430)
        left_v = QVBoxLayout(left_col)
        left_v.setContentsMargins(0, 0, 0, 0)
        left_v.addWidget(config_scroll, 1)

        run_row = QHBoxLayout()
        self.train_run_btn = QPushButton("Run LOSO Training")
        self.train_run_btn.clicked.connect(self._run_training)
        self.train_cancel_btn = QPushButton("Cancel")
        self.train_cancel_btn.clicked.connect(self._cancel_training)
        self.train_cancel_btn.setEnabled(False)
        run_row.addWidget(self.train_run_btn)
        run_row.addWidget(self.train_cancel_btn)
        left_v.addLayout(run_row)

        self.train_progress_label = QLabel("Idle")
        self.train_progress_label.setWordWrap(True)
        left_v.addWidget(self.train_progress_label)

        outer.addWidget(left_col)

        # --- Right: log, results table, scatter plot ---
        right = QWidget()
        rl = QVBoxLayout(right)

        right_splitter = QSplitter(Qt.Vertical)
        rl.addWidget(right_splitter)

        self.train_log_edit = QTextEdit()
        self.train_log_edit.setReadOnly(True)
        self.train_log_edit.setFont(QFont("Monospace", 9))
        right_splitter.addWidget(self.train_log_edit)

        results_widget = QWidget()
        results_layout = QVBoxLayout(results_widget)
        results_layout.addWidget(QLabel("Per-subject LOSO results (base model, whole subject, MAE mmHg):"))
        self.train_results_table = QTableView()
        self.train_results_model = PandasModel(pd.DataFrame())
        self.train_results_table.setModel(self.train_results_model)
        results_layout.addWidget(self.train_results_table)
        export_results_btn = QPushButton("Export summary CSV...")
        export_results_btn.clicked.connect(self._export_training_summary)
        results_layout.addWidget(export_results_btn)

        # Second table: personalized (fine-tuned) results on each subject's own
        # held-out test slice. Only populated when 'Personalize' is on. Kept
        # separate because these are on a different (smaller) slice than the base
        # table above - base->post is only fair against p_base (same slice).
        self.train_personalized_label = QLabel(
            "Per-subject PERSONALIZED results (fine-tuned, held-out slice; "
            "compare post vs base vs zero-delta, MAE mmHg):")
        results_layout.addWidget(self.train_personalized_label)
        self.train_personalized_table = QTableView()
        self.train_personalized_model = PandasModel(pd.DataFrame())
        self.train_personalized_table.setModel(self.train_personalized_model)
        results_layout.addWidget(self.train_personalized_table)
        self.train_personalized_export_btn = QPushButton("Export personalized summary CSV...")
        self.train_personalized_export_btn.clicked.connect(self._export_personalized_summary)
        results_layout.addWidget(self.train_personalized_export_btn)
        # Hidden until a run actually produces personalized rows.
        self.train_personalized_label.setVisible(False)
        self.train_personalized_table.setVisible(False)
        self.train_personalized_export_btn.setVisible(False)
        right_splitter.addWidget(results_widget)

        loss_widget = QWidget()
        loss_layout = QVBoxLayout(loss_widget)
        loss_top = QHBoxLayout()
        loss_top.addWidget(QLabel("Train/val loss curve for fold (test subject):"))
        self.loss_fold_combo = QComboBox()
        self.loss_fold_combo.currentTextChanged.connect(self._update_loss_plot)
        loss_top.addWidget(self.loss_fold_combo)
        loss_top.addStretch(1)
        loss_layout.addLayout(loss_top)
        self.loss_canvas = MplCanvas(width=8, height=3)
        loss_layout.addWidget(self.loss_canvas)
        right_splitter.addWidget(loss_widget)

        self.train_scatter_canvas = MplCanvas(width=8, height=4)
        right_splitter.addWidget(self.train_scatter_canvas)

        outer.addWidget(right, stretch=1)
        self.tabs.addTab(tab, "Model Training")

    def _build_test_tab(self):
        tab = QWidget()
        outer = QHBoxLayout(tab)

        if _TRAIN_WORKER_IMPORT_ERROR is not None:
            err_label = QLabel(
                "Could not load the testing module (PyTorch failed to import). "
                "See the Model Training tab for details."
            )
            err_label.setWordWrap(True)
            err_label.setStyleSheet("color: #b00020; padding: 20px;")
            outer.addWidget(err_label)
            self.tabs.addTab(tab, "Test Saved Model")
            return

        control_panel = QWidget()
        control_panel.setMaximumWidth(360)
        cl = QVBoxLayout(control_panel)

        ckpt_group = QGroupBox("Trained model")
        ckpt_form = QFormLayout(ckpt_group)
        self.test_ckpt_edit = QLineEdit()
        ckpt_btn = QPushButton("Browse...")
        ckpt_btn.clicked.connect(self._browse_test_checkpoint)
        ckpt_row = QHBoxLayout()
        ckpt_row.addWidget(self.test_ckpt_edit)
        ckpt_row.addWidget(ckpt_btn)
        ckpt_form.addRow("checkpoint.pt:", ckpt_row)
        ckpt_hint = QLabel("Any fold_<subject>/checkpoint.pt saved by LOSO training,\n"
                            "or any checkpoint you saved yourself in the same format.")
        ckpt_hint.setStyleSheet("color: gray; font-size: 10px;")
        ckpt_form.addRow("", ckpt_hint)
        cl.addWidget(ckpt_group)

        data_group = QGroupBox("Dataset to test on")
        data_form = QFormLayout(data_group)
        self.test_data_dir_edit = QLineEdit()
        test_data_btn = QPushButton("Browse...")
        test_data_btn.clicked.connect(self._browse_test_data_dir)
        test_data_row = QHBoxLayout()
        test_data_row.addWidget(self.test_data_dir_edit)
        test_data_row.addWidget(test_data_btn)
        data_form.addRow("Combined dir:", test_data_row)
        data_hint = QLabel("Can be the same combined dir used for training (to inspect\n"
                            "a specific held-out subject), or a brand-new aggregated\n"
                            "dataset from recordings collected later.")
        data_hint.setStyleSheet("color: gray; font-size: 10px;")
        data_form.addRow("", data_hint)

        self.test_subjects_edit = QLineEdit()
        self.test_subjects_edit.setPlaceholderText("blank = use every subject in this dataset")
        data_form.addRow("Subject filter:", self.test_subjects_edit)
        subj_hint = QLabel("Comma-separated Subject_IDs, e.g. s5,s6 - leave blank to test on all.\n"
                            "A leakage guard auto-drops any subject this checkpoint's own fold\n"
                            "already trained on, so 'blank = all' only ever scores the true\n"
                            "held-out subject (plus any brand-new, never-trained-on recording).")
        subj_hint.setStyleSheet("color: gray; font-size: 10px;")
        data_form.addRow("", subj_hint)
        cl.addWidget(data_group)

        self.test_use_gpu_check = QCheckBox("Use GPU if available")
        self.test_use_gpu_check.setChecked(True)
        cl.addWidget(self.test_use_gpu_check)

        self.test_allow_in_sample_check = QCheckBox(
            "Include in-sample subjects anyway (debug only - inflates metrics)")
        self.test_allow_in_sample_check.setChecked(False)
        cl.addWidget(self.test_allow_in_sample_check)

        self.test_run_btn = QPushButton("Run Test")
        self.test_run_btn.clicked.connect(self._run_test)
        cl.addWidget(self.test_run_btn)

        self.test_metrics_label = QLabel("No results yet.")
        self.test_metrics_label.setWordWrap(True)
        cl.addWidget(self.test_metrics_label)

        export_pred_btn = QPushButton("Export predictions CSV...")
        export_pred_btn.clicked.connect(self._export_test_predictions)
        cl.addWidget(export_pred_btn)

        cl.addStretch(1)
        outer.addWidget(control_panel)

        right = QWidget()
        rl = QVBoxLayout(right)
        self.test_log_edit = QTextEdit()
        self.test_log_edit.setReadOnly(True)
        self.test_log_edit.setFont(QFont("Monospace", 9))
        self.test_log_edit.setMaximumHeight(150)
        rl.addWidget(self.test_log_edit)
        self.test_scatter_canvas = MplCanvas(width=8, height=5)
        rl.addWidget(self.test_scatter_canvas)
        outer.addWidget(right, stretch=1)

        self.tabs.addTab(tab, "Test Saved Model")

    def _browse_test_checkpoint(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select checkpoint.pt", "", "PyTorch checkpoint (*.pt)")
        if path:
            self.test_ckpt_edit.setText(path)

    def _browse_test_data_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Select the aggregated dataset directory to test on")
        if path:
            self.test_data_dir_edit.setText(path)

    def _run_test(self):
        ckpt_path = self.test_ckpt_edit.text().strip()
        data_dir = self.test_data_dir_edit.text().strip()
        if not ckpt_path or not os.path.isfile(ckpt_path):
            QMessageBox.warning(self, "Missing checkpoint", "Please select a valid checkpoint.pt file.")
            return
        required = ["all_raw_signal.npz", "all_raw_signal_labels.csv", "all_skna_signal.npz",
                    "all_iskna_rskna.npz", "all_askna.npz"]
        if not data_dir or not os.path.isdir(data_dir):
            QMessageBox.warning(self, "Missing directory", "Please select a valid aggregated dataset directory.")
            return
        missing = [f for f in required if not os.path.isfile(os.path.join(data_dir, f))]
        if missing:
            QMessageBox.warning(self, "Missing files",
                                 "The selected directory is missing:\n" + "\n".join(missing))
            return

        subjects_text = self.test_subjects_edit.text().strip()
        test_subjects = [s.strip() for s in subjects_text.split(",") if s.strip()] if subjects_text else None

        params = dict(
            checkpoint_path=ckpt_path,
            combined_dir=data_dir,
            test_subjects=test_subjects,
            use_gpu=self.test_use_gpu_check.isChecked(),
            allow_in_sample=self.test_allow_in_sample_check.isChecked(),
        )

        self.test_log_edit.clear()
        self.test_metrics_label.setText("Running...")
        self.test_run_btn.setEnabled(False)

        self.test_worker = InferenceWorker(params)
        self.test_worker.log.connect(lambda m: self.test_log_edit.append(m))
        self.test_worker.error.connect(self._on_test_error)
        self.test_worker.finished_ok.connect(self._on_test_finished)
        self.test_worker.start()

    def _on_test_error(self, msg):
        self.test_run_btn.setEnabled(True)
        self.test_metrics_label.setText("Error - see log.")
        self.test_log_edit.append("ERROR:\n" + msg)
        QMessageBox.critical(self, "Test error", msg.split("\n\n")[0])

    def _on_test_finished(self, result):
        self.test_run_btn.setEnabled(True)
        metrics = result["metrics"]
        self.test_predictions_df = result["predictions"]
        self.test_metrics_label.setText(
            f"MAE_SBP = {metrics['MAE_SBP']:.2f} mmHg\n"
            f"MAE_DBP = {metrics['MAE_DBP']:.2f} mmHg\n"
            f"ME +/- SD SBP = {metrics['bias_SBP']:+.2f} +/- {metrics['SDE_SBP']:.2f} mmHg\n"
            f"ME +/- SD DBP = {metrics['bias_DBP']:+.2f} +/- {metrics['SDE_DBP']:.2f} mmHg\n"
            f"corr_SBP (r) = {metrics['corr_SBP']:.3f}\n"
            f"corr_DBP (r) = {metrics['corr_DBP']:.3f}\n"
            f"n = {len(self.test_predictions_df)} sequences"
        )

        df = self.test_predictions_df
        fig = self.test_scatter_canvas.fig
        fig.clear()
        ax1, ax2 = fig.subplots(1, 2)
        for ax, true_col, pred_col, title in [
            (ax1, "SBP_true", "SBP_pred", "SBP: predicted vs true"),
            (ax2, "DBP_true", "DBP_pred", "DBP: predicted vs true"),
        ]:
            ax.scatter(df[true_col], df[pred_col], s=8, alpha=0.4)
            # Both axes anchored at 0 with a shared [0, hi] range so the y=x line
            # stays a true 45-degree reference.
            hi = max(df[true_col].max(), df[pred_col].max()) * 1.05
            ax.plot([0, hi], [0, hi], 'r--', linewidth=1, label="y = x")
            ax.set_xlim(0, hi)
            ax.set_ylim(0, hi)
            ax.set_xlabel("True (mmHg)")
            ax.set_ylabel("Predicted (mmHg)")
            ax.set_title(title)
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.4)
        fig.tight_layout()
        self.test_scatter_canvas.draw()

    def _export_test_predictions(self):
        df = getattr(self, "test_predictions_df", None)
        if df is None or df.empty:
            QMessageBox.warning(self, "Nothing to export", "Run a test first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export predictions CSV", "test_predictions.csv", "CSV files (*.csv)")
        if path:
            df.to_csv(path, index=False)
            self.test_log_edit.append(f"Exported predictions -> {path}")

    def _browse_txt(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select SKNA/ECG .txt file", "", "Text files (*.txt);;All files (*)")
        if path:
            self.txt_edit.setText(path)

    def _browse_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select vitals .csv file", "", "CSV files (*.csv);;All files (*)")
        if path:
            self.csv_edit.setText(path)

    def _browse_out(self):
        path = QFileDialog.getExistingDirectory(self, "Select output directory")
        if path:
            self.out_edit.setText(path)

    def _run_pipeline(self):
        txt_path = self.txt_edit.text().strip()
        csv_path = self.csv_edit.text().strip()
        out_dir = self.out_edit.text().strip()

        if not txt_path or not os.path.isfile(txt_path):
            QMessageBox.warning(self, "Missing file", "Please select a valid SKNA/ECG .txt file.")
            return
        if not csv_path or not os.path.isfile(csv_path):
            QMessageBox.warning(self, "Missing file", "Please select a valid vitals .csv file.")
            return
        subject_id = self.subject_id_edit.text().strip()
        if not subject_id:
            QMessageBox.warning(self, "Missing Subject ID",
                                 "Please enter a Subject ID (e.g. S01) so this recording's "
                                 "output can be told apart from other subjects later.")
            return
        os.makedirs(out_dir, exist_ok=True)

        params = dict(
            txt_filename=txt_path,
            csv_filename=csv_path,
            subject_id=subject_id,
            session_id=self.session_id_edit.text().strip(),
            fs=self.fs_spin.value(),
            window_sec=self.window_sec_spin.value(),
            hpf_enabled=self.hpf_check.isChecked(),
            hpf_fc=self.hpf_fc_spin.value(),
            hpf_order=self.hpf_order_spin.value(),
            output_dir=out_dir,
            detrend_full_signal=self.detrend_full_check.isChecked(),
            detrend_full_type=self.detrend_full_type_combo.currentText(),
            detrend_signals=self.detrend_window_check.isChecked(),
            detrend_type=self.detrend_window_type_combo.currentText(),
            output_format="npz" if self.npz_output_check.isChecked() else "csv",
            run_cwt=self.cwt_check.isChecked(),
            save_cwt_images=self.cwt_save_images_check.isChecked(),
            cwt_window_sec=self.cwt_window_spin.value(),
            compute_cwt_arrays=self.cwt_arrays_check.isChecked(),
            cwt_time_bins=self.cwt_time_bins_spin.value(),
            compute_fd_arrays=self.fd_arrays_check.isChecked(),
        )

        self.log_edit.clear()
        self.progress_bar.setValue(0)
        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)

        self.worker = PipelineWorker(params)
        self.worker.log.connect(self._append_log)
        self.worker.progress.connect(self._on_progress)
        self.worker.stage_started.connect(lambda name: self.progress_label.setText(f"Running: {name}"))
        self.worker.stage_finished.connect(self._on_stage_finished)
        self.worker.error.connect(self._on_error)
        self.worker.finished_all.connect(self._on_finished_all)
        self.worker.start()

    def _cancel_pipeline(self):
        if self.worker and self.worker.isRunning():
            self.worker.request_stop()
            self._append_log("Cancel requested...")

    def _append_log(self, msg):
        self.log_edit.append(msg)

    def _on_progress(self, stage, cur, total):
        self.progress_label.setText(f"{stage}: {cur}/{total}")
        pct = int(100 * cur / max(total, 1))
        self.progress_bar.setValue(pct)

    def _on_error(self, msg):
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        if msg == "__CANCELLED__":
            self.progress_label.setText("Cancelled")
            return
        self.progress_label.setText("Error")
        self._append_log("ERROR:\n" + msg)
        QMessageBox.critical(self, "Pipeline error", msg.split("\n\n")[0])

    def _on_stage_finished(self, stage, payload):
        self.results.update(payload)

        if stage == "load":
            self._plot_raw_channels(payload["df_txt"])
            self._plot_vitals()

        elif stage == "detrend":
            self._plot_raw_channels(payload["df_txt"])

        elif stage == "preprocess":
            self._plot_filtered_signals(payload["skna_filtered"], payload["rskna"], payload["askna"])

        elif stage == "window":
            n_windows = len(payload["skna_windows"])
            self.window_spin.setMaximum(max(n_windows - 1, 0))
            self.window_count_label.setText(f"of {n_windows - 1}")
            self._update_window_view()

        elif stage == "features":
            self._populate_feature_tables(payload)

        elif stage == "cwt":
            self._populate_feature_tables(payload)

        elif stage == "phase_stats":
            self._plot_phase_comparison(payload)

    def _on_finished_all(self, results):
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.progress_label.setText("Done")
        self.progress_bar.setValue(100)

    def _plot_raw_channels(self, df_txt):
        # The plotted channels are min-max normalized to [0, 1] for overlay, but
        # otherwise reflect exactly what feeds windowing/training: high-pass
        # filtered (if the HPF is on) and, at the detrend stage, detrended too.
        self.plot_raw.clear()
        t = df_txt['Time_min'].to_numpy()
        colors = {'CH1': '#d62728', 'CH40': '#1f77b4', 'CH41': '#2ca02c'}
        has_detrended = any(f"{ch}_detrended" in df_txt.columns for ch in colors)
        for ch, color in colors.items():
            if ch not in df_txt.columns:
                continue
            detrended_col = f"{ch}_detrended"
            if detrended_col in df_txt.columns:
                y = df_txt[detrended_col].to_numpy()
            else:
                y = core.normalize_minmax(df_txt[ch].to_numpy())

            x, y = downsample_for_plot(t, y)
            self.plot_raw.plot(x, y, pen=pg.mkPen(color, width=1.5), name=ch)

        # Title states what is actually shown, so it's clear the plot is the
        # HPF result (the trained signal) and not the untouched raw import.
        hpf_on = getattr(self, "hpf_check", None) is not None and self.hpf_check.isChecked()
        stages = [s for s, on in (("HPF", hpf_on), ("detrended", has_detrended)) if on]
        prefix = " + ".join(stages) if stages else "Raw"
        self.plot_raw.setTitle(f"{prefix} channels (CH1 / CH40 / CH41)")

    def _plot_filtered_signals(self, skna_filtered, rskna, askna):
        fs = self.fs_spin.value()
        window_sec = self.window_sec_spin.value()

        # Share the raw-channel plot's (alignment-shifted) time base so all five
        # X-linked panels line up. skna_filtered comes from the same samples as
        # df_txt, so they are the same length; fall back to a plain 0-based axis
        # if df_txt isn't available yet.
        df_txt = self.results.get("df_txt")
        if (df_txt is not None and 'Time_min' in df_txt
                and len(df_txt) == len(skna_filtered)):
            t_full_min = df_txt['Time_min'].to_numpy()
            t0_min = float(t_full_min[0]) if len(t_full_min) else 0.0
        else:
            t_full_min = np.arange(len(skna_filtered)) / fs / 60.0
            t0_min = 0.0

        self.plot_filtered.clear()
        x, y = downsample_for_plot(t_full_min, skna_filtered)
        self.plot_filtered.plot(x, y, pen=pg.mkPen('k', width=1))

        self.plot_rskna.clear()
        x, y = downsample_for_plot(t_full_min, rskna)
        self.plot_rskna.plot(x, y, pen=pg.mkPen('b', width=1))

        self.plot_askna.clear()
        # One aSKNA point per window, placed at the window start in the same base.
        t_askna_min = t0_min + (np.arange(len(askna)) * window_sec) / 60.0
        self.plot_askna.setTitle(f"aSKNA ({window_sec}s average)")
        self.plot_askna.plot(t_askna_min, askna, pen=pg.mkPen('r', width=2), symbol='o', symbolSize=4)

        self._plot_vitals()

    def _plot_vitals(self):
        df_csv = self.results.get("df_csv")
        if df_csv is None:
            return
        self.plot_vitals.clear()
        vital_cols = [c for c in df_csv.columns if any(t in c.upper() for t in ['SYS', 'DIA', 'MAP'])]
        colors = ['b', 'g', 'purple', 'orange', '#17a2a2', 'm']
        for i, col in enumerate(vital_cols):
            self.plot_vitals.plot(df_csv['Time_min'].to_numpy(), df_csv[col].to_numpy(),
                                   pen=pg.mkPen(colors[i % len(colors)], width=1.5), name=col)

    def _update_window_view(self):
        skna_windows = self.results.get("skna_windows")
        if skna_windows is None or len(skna_windows) == 0:
            return
        i = min(self.window_spin.value(), len(skna_windows) - 1)
        fs = self.results.get("fs", self.fs_spin.value())
        window_size = self.results.get("window_size", skna_windows.shape[1])

        w_skna = skna_windows[i]
        w_rskna = self.results["rskna_windows"][i]
        w_iskna = self.results["iskna_windows"][i]
        t = np.arange(window_size) / fs

        self.win_plot_skna.clear()
        self.win_plot_skna.plot(t, w_skna, pen=pg.mkPen('k'))
        self.win_plot_rskna.clear()
        self.win_plot_rskna.plot(t, w_rskna, pen=pg.mkPen('b'))
        self.win_plot_iskna.clear()
        self.win_plot_iskna.plot(t, w_iskna, pen=pg.mkPen('g'))

        # DC-removed (w_skna sits on a ~0.5 baseline post-normalization; see
        # extract_fft_features) then rescaled to [0, 1] via plain linear
        # min-max (see core.normalize_minmax) - matches compute_fd_array's
        # normalization (the actual FD training-array builder), NOT
        # extract_fft_features/extract_psd_features (the summary-feature
        # functions), which still deliberately use normalize_log_minmax -
        # see [[feedback_spectral_normalization]] in memory for why those
        # two paths now intentionally differ.
        f_axis = rfftfreq(window_size, d=1 / fs)
        mag = core.normalize_minmax(np.abs(rfft(w_skna - np.mean(w_skna))))
        self.win_plot_fft.clear()
        self.win_plot_fft.plot(f_axis, mag, pen=pg.mkPen('m'))

        f_psd, psd = welch(w_skna, fs, nperseg=window_size, detrend='constant')
        psd = core.normalize_minmax(psd)
        self.win_plot_psd.clear()
        self.win_plot_psd.plot(f_psd, psd, pen=pg.mkPen('r'))

    def _compute_cwt_preview(self):
        skna_windows = self.results.get("skna_windows")
        if skna_windows is None:
            return
        i = min(self.window_spin.value(), len(skna_windows) - 1)
        fs = self.results.get("fs", self.fs_spin.value())
        window = skna_windows[i]

        self.progress_label.setText("Computing CWT preview...")
        QApplication.processEvents()
        mag, frequencies = core.compute_single_cwt(window, fs)
        # Log-compressed + robustly normalized to [0, 1] (see
        # core.normalize_log_minmax), same as extract_cwt_features, so this
        # preview matches what actually gets saved/fed to the model.
        mag = core.normalize_log_minmax(mag)

        self.cwt_canvas.fig.clear()
        ax = self.cwt_canvas.fig.add_subplot(111)
        im = ax.imshow(mag, aspect='auto',
                        extent=[0, len(window) / fs, frequencies[0], frequencies[-1]],
                        origin='lower', cmap='gray', vmin=0, vmax=1)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title(f"CWT Scalogram - Window {i}")
        self.cwt_canvas.fig.colorbar(im, ax=ax, label="Normalized magnitude")
        self.cwt_canvas.fig.tight_layout()
        self.cwt_canvas.draw()
        self.progress_label.setText("Idle")

    def _populate_feature_tables(self, payload):
        for name, df in payload.items():
            if not isinstance(df, pd.DataFrame):
                continue
            if name not in self.feature_models:
                self.feature_models[name] = PandasModel(df)
                self.feature_combo.addItem(name)
            else:
                self.feature_models[name].set_df(df)

        if self.feature_combo.count() > 0 and self.feature_table.model() is None:
            self.feature_combo.setCurrentIndex(0)
            self._show_feature_table(self.feature_combo.currentText())

    def _show_feature_table(self, name):
        model = self.feature_models.get(name)
        if model is not None:
            self.feature_table.setModel(model)

    def _export_current_feature(self):
        name = self.feature_combo.currentText()
        model = self.feature_models.get(name)
        if model is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export CSV", f"{name}.csv", "CSV files (*.csv)")
        if path:
            model._df.to_csv(path, index=False)
            self._append_log(f"Exported {name} -> {path}")

    def _plot_phase_comparison(self, payload):
        askna_stats = payload["askna_stats"]
        vitals_stats = payload["vitals_stats"]
        df_askna = payload["df_askna"]
        df_vitals = payload["df_vitals"]

        try:
            core.fig_statistical_comparison(askna_stats, vitals_stats, fig=self.phase_bar_canvas.fig)
            self.phase_bar_canvas.draw()
        except Exception as e:
            self._append_log(f"Could not build statistical comparison figure: {e}")

        try:
            core.fig_continuous_segmented_signals(df_askna, df_vitals, fig=self.phase_series_canvas.fig)
            self.phase_series_canvas.draw()
        except Exception as e:
            self._append_log(f"Could not build segmented time series figure: {e}")

    def _save_phase_figures(self):
        out_dir = self.out_edit.text().strip() or "."
        os.makedirs(out_dir, exist_ok=True)
        fig1 = self.phase_bar_canvas.fig if self.phase_bar_canvas.fig.axes else None
        fig2 = self.phase_series_canvas.fig if self.phase_series_canvas.fig.axes else None
        saved = []
        if fig1 is not None:
            path1 = os.path.join(out_dir, "askna_bp_meanerr.png")
            fig1.savefig(path1, dpi=300)
            saved.append(path1)
        if fig2 is not None:
            path2 = os.path.join(out_dir, "askna_bp_comparison.png")
            fig2.savefig(path2, dpi=300)
            saved.append(path2)
        if saved:
            QMessageBox.information(self, "Saved", "Saved:\n" + "\n".join(saved))
        else:
            QMessageBox.warning(self, "Nothing to save", "Run the pipeline first.")

    # ---------------- Model Training tab ----------------

    def _browse_combined_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Select the aggregated ('combined') dataset directory")
        if path:
            self.combined_dir_edit.setText(path)

    def _on_val_mode_changed(self):
        self.val_frac_spin.setEnabled(self.val_mode_combo.currentData() == "time_split")

    def _browse_train_out(self):
        path = QFileDialog.getExistingDirectory(self, "Select directory for checkpoints/predictions")
        if path:
            self.train_out_edit.setText(path)

    def _run_training(self):
        combined_dir = self.combined_dir_edit.text().strip()
        required = ["all_raw_signal.npz", "all_raw_signal_labels.csv", "all_skna_signal.npz",
                    "all_iskna_rskna.npz", "all_askna.npz"]
        # Optional-branch inputs are only required when their branch is enabled.
        if self.use_fft_check.isChecked() or self.use_psd_check.isChecked():
            required.append("all_fd_signal.npz")
        if self.use_cwt_check.isChecked():
            required.append("all_cwt_signal.npz")
        if not combined_dir or not os.path.isdir(combined_dir):
            QMessageBox.warning(self, "Missing directory", "Please select a valid aggregated dataset directory.")
            return
        missing = [f for f in required if not os.path.isfile(os.path.join(combined_dir, f))]
        if missing:
            QMessageBox.warning(self, "Missing files",
                                 "The selected directory is missing:\n" + "\n".join(missing) +
                                 "\n\nRun aggregate_datasets.py --format npz first.")
            return

        out_dir = self.train_out_edit.text().strip() or "training_results"
        os.makedirs(out_dir, exist_ok=True)

        params = dict(
            combined_dir=combined_dir,
            output_dir=out_dir,
            seq_len=self.seq_len_spin.value(),
            stride=self.stride_spin.value(),
            downsample=self.downsample_spin.value(),
            val_mode=self.val_mode_combo.currentData(),
            val_frac=self.val_frac_spin.value(),
            norm_mode=self.norm_mode_combo.currentData(),
            exclude_subjects=[s.strip() for s in self.exclude_subjects_edit.text().split(",") if s.strip()],
            batch_size=self.batch_size_spin.value(),
            epochs=self.epochs_spin.value(),
            patience=self.patience_spin.value(),
            lr=self.lr_spin.value(),
            weight_decay=self.weight_decay_spin.value(),
            lstm_hidden=self.lstm_hidden_spin.value(),
            lstm_layers=self.lstm_layers_spin.value(),
            # CNN branch depth: channels double per block from 16 (2->16,32; 3->16,32,64).
            cnn_channels=tuple(16 * (2 ** i) for i in range(self.cnn_layers_spin.value())),
            cnn_arch=self.cnn_arch_combo.currentData(),
            cwt_arch=self.cwt_arch_combo.currentData(),
            ann_pool_len=self.ann_pool_len_spin.value(),
            dropout=self.dropout_spin.value(),
            anti_collapse_lambda=self.anti_collapse_lambda_spin.value(),
            use_ecg=self.use_ecg_check.isChecked(),
            use_skna=self.use_skna_check.isChecked(),
            use_iskna_rskna=self.use_iskna_rskna_check.isChecked(),
            use_askna=self.use_askna_check.isChecked(),
            use_fft=self.use_fft_check.isChecked(),
            use_psd=self.use_psd_check.isChecked(),
            use_cwt=self.use_cwt_check.isChecked(),
            use_gpu=self.use_gpu_check.isChecked(),
            personalize=self.personalize_check.isChecked(),
            adapt_end=self.adapt_end_spin.value(),
            val_end=self.val_end_spin.value(),
            ft_lr=self.ft_lr_spin.value(),
            ft_epochs=self.ft_epochs_spin.value(),
            ft_patience=self.ft_patience_spin.value(),
        )

        if params["personalize"] and params["val_end"] <= params["adapt_end"]:
            QMessageBox.warning(
                self, "Invalid fine-tune split",
                f"FT val split end ({params['val_end']:.2f}) must be greater than "
                f"FT adapt split end ({params['adapt_end']:.2f}) so the adapt, val, "
                f"and held-out test slices don't overlap.")
            self.train_run_btn.setEnabled(True)
            self.train_cancel_btn.setEnabled(False)
            return

        if not any(params[k] for k in ("use_ecg", "use_skna", "use_iskna_rskna",
                                       "use_askna", "use_fft", "use_psd", "use_cwt")):
            QMessageBox.warning(self, "No CNN inputs",
                                "Enable at least one CNN input signal before training.")
            self.train_run_btn.setEnabled(True)
            self.train_cancel_btn.setEnabled(False)
            return

        self.train_log_edit.clear()
        self.fold_predictions = {}
        self.fold_personalized = {}
        self.fold_loss_history = {}
        self.loss_fold_combo.blockSignals(True)
        self.loss_fold_combo.clear()
        self.loss_fold_combo.blockSignals(False)
        self.loss_canvas.fig.clear()
        self.loss_canvas.draw()
        self.train_results_model.set_df(pd.DataFrame())
        self.train_personalized_model.set_df(pd.DataFrame())
        self.train_personalized_label.setVisible(False)
        self.train_personalized_table.setVisible(False)
        self.train_personalized_export_btn.setVisible(False)
        self.train_scatter_canvas.fig.clear()
        self.train_scatter_canvas.draw()
        self.train_run_btn.setEnabled(False)
        self.train_cancel_btn.setEnabled(True)
        self.train_progress_label.setText("Running...")

        self.train_worker = LOSOTrainingWorker(params)
        self.train_worker.log.connect(self._train_append_log)
        self.train_worker.fold_started.connect(
            lambda subj: self.train_progress_label.setText(f"Training fold: test subject = {subj}"))
        self.train_worker.epoch_progress.connect(self._on_train_epoch_progress)
        self.train_worker.fold_finished.connect(self._on_train_fold_finished)
        self.train_worker.error.connect(self._on_train_error)
        self.train_worker.all_finished.connect(self._on_train_all_finished)
        self.train_worker.start()

    def _cancel_training(self):
        if self.train_worker and self.train_worker.isRunning():
            self.train_worker.request_stop()
            self._train_append_log("Cancel requested...")

    def _train_append_log(self, msg):
        self.train_log_edit.append(msg)

    # preset key -> (ecg, skna, iskna_rskna, askna, fft, psd, cwt)
    _INPUT_PRESETS = {
        "all":       (True,  True,  True,  True,  False, False, False),
        "ecg_only":  (True,  False, False, False, False, False, False),
        "skna_only": (False, True,  True,  True,  False, False, False),
        "ecg_skna":  (True,  True,  False, False, False, False, False),
    }

    def _apply_input_preset(self):
        """Preset dropdown changed -> tick the input checkboxes to match."""
        key = self.input_preset_combo.currentData()
        cfg = self._INPUT_PRESETS.get(key)
        if cfg is None:   # "custom" - leave the boxes as the user set them
            return
        self._applying_preset = True
        try:
            for cb, on in zip(self._input_checks, cfg):
                cb.setChecked(on)
        finally:
            self._applying_preset = False

    def _mark_input_preset_custom(self):
        """A checkbox was hand-edited -> switch the preset dropdown to 'Custom'."""
        if self._applying_preset:
            return
        idx = self.input_preset_combo.findData("custom")
        if idx >= 0 and self.input_preset_combo.currentIndex() != idx:
            self.input_preset_combo.blockSignals(True)
            self.input_preset_combo.setCurrentIndex(idx)
            self.input_preset_combo.blockSignals(False)

    def _on_train_epoch_progress(self, subject, epoch, train_loss, val_loss, val_select):
        self.train_progress_label.setText(
            f"Fold {subject}: epoch {epoch} | train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} | val_MAE_SBP+DBP={val_select:.3f} mmHg")

        hist = self.fold_loss_history.setdefault(
            subject, {"epoch": [], "train_loss": [], "val_loss": [], "val_select": []})
        hist["epoch"].append(epoch)
        hist["train_loss"].append(train_loss)
        hist["val_loss"].append(val_loss)
        hist["val_select"].append(val_select)

        if self.loss_fold_combo.findText(subject) < 0:
            self.loss_fold_combo.addItem(subject)
            self.loss_fold_combo.setCurrentText(subject)  # auto-follow the newest fold live
        # Live-update the plot only while viewing the fold currently training
        # (or if nothing's been selected yet), so switching to review a
        # finished fold isn't interrupted by the next fold's progress.
        if self.loss_fold_combo.currentText() == subject:
            self._update_loss_plot(subject)

    def _update_loss_plot(self, subject):
        if not subject or subject not in self.fold_loss_history:
            return
        hist = self.fold_loss_history[subject]
        fig = self.loss_canvas.fig
        fig.clear()
        ax = fig.add_subplot(111)
        # Training and validation loss share one axis: both are the Huber loss on
        # the same scaler-normalized delta target, so they are directly comparable
        # within a fold and a train/val gap reads as genuine overfitting. (The
        # early-stopping metric, val MAE in mmHg, lives in the progress label and
        # the results table - it isn't plotted here so the loss curves stay on a
        # single, comparable scale.)
        ax.plot(hist["epoch"], hist["train_loss"], label="training loss",
                marker='o', markersize=3, color="tab:blue")
        ax.plot(hist["epoch"], hist.get("val_loss", []), label="validation loss",
                marker='o', markersize=3, color="tab:orange")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (Huber, normalized target)")
        ax.set_title(f"Fold test={subject}: training vs validation loss")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.4)
        fig.tight_layout()
        self.loss_canvas.draw()

    def _on_train_error(self, msg):
        self.train_run_btn.setEnabled(True)
        self.train_cancel_btn.setEnabled(False)
        if msg == "__CANCELLED__":
            self.train_progress_label.setText("Cancelled")
            return
        self.train_progress_label.setText("Error")
        self._train_append_log("ERROR:\n" + msg)
        QMessageBox.critical(self, "Training error", msg.split("\n\n")[0])

    def _on_personalize_toggled(self, checked):
        # The fine-tune knobs only matter when Personalize is on; grey them out
        # otherwise so it's clear they're inert (the worker uses its defaults).
        for spin in self._ft_knob_spins:
            spin.setEnabled(checked)

    def _on_train_fold_finished(self, subject, fold_result):
        self.fold_predictions[subject] = fold_result["predictions"]

        summary_rows = []
        for subj, df in self.fold_predictions.items():
            err_sbp = df["SBP_pred"] - df["SBP_true"]
            err_dbp = df["DBP_pred"] - df["DBP_true"]
            mae_sbp = err_sbp.abs().mean()
            mae_dbp = err_dbp.abs().mean()
            # ME +/- SD: signed mean error and its spread (AAMI/ISO error pair).
            me_sbp, sd_sbp = err_sbp.mean(), err_sbp.std(ddof=1)
            me_dbp, sd_dbp = err_dbp.mean(), err_dbp.std(ddof=1)
            corr_sbp = df["SBP_pred"].corr(df["SBP_true"]) if df["SBP_true"].std() > 1e-8 else float('nan')
            corr_dbp = df["DBP_pred"].corr(df["DBP_true"]) if df["DBP_true"].std() > 1e-8 else float('nan')
            pred_std_sbp = df["SBP_pred"].std()
            true_std_sbp = df["SBP_true"].std()
            summary_rows.append({
                "Subject_ID": subj, "MAE_SBP": mae_sbp, "MAE_DBP": mae_dbp,
                "ME_SBP": me_sbp, "SD_SBP": sd_sbp, "ME_DBP": me_dbp, "SD_DBP": sd_dbp,
                "corr_SBP": corr_sbp, "corr_DBP": corr_dbp,
                "pred_std_SBP": pred_std_sbp, "true_std_SBP": true_std_sbp,
                "n_test_seq": len(df),
            })
        summary_df = pd.DataFrame(summary_rows)
        self.train_results_model.set_df(summary_df)
        self._update_train_scatter()

        # If this fold produced personalized (fine-tuned) results, add its row to
        # the second table. Metrics are on that subject's own held-out slice.
        if fold_result.get("p_predictions") is not None:
            self.fold_personalized[subject] = fold_result
            self._update_personalized_table()

    def _update_personalized_table(self):
        rows = []
        for subj, fr in self.fold_personalized.items():
            base_sbp, base_dbp = fr.get("p_base_MAE_SBP"), fr.get("p_base_MAE_DBP")
            post_sbp, post_dbp = fr.get("p_post_MAE_SBP"), fr.get("p_post_MAE_DBP")
            bl_sbp, bl_dbp = fr.get("p_baseline_MAE_SBP"), fr.get("p_baseline_MAE_DBP")
            rows.append({
                "Subject_ID": subj, "n_test": fr.get("p_n_test"),
                "base_MAE_SBP": base_sbp, "post_MAE_SBP": post_sbp,
                "zerodelta_SBP": bl_sbp,
                "base_MAE_DBP": base_dbp, "post_MAE_DBP": post_dbp,
                "zerodelta_DBP": bl_dbp,
                # Did fine-tuning actually help vs the same-slice base model?
                "post<base": (post_sbp is not None and base_sbp is not None
                              and post_sbp < base_sbp and post_dbp < base_dbp),
                "beats_zerodelta": fr.get("p_beats_baseline"),
                "corr_SBP_delta": fr.get("p_corr_SBP_delta"),
                "corr_DBP_delta": fr.get("p_corr_DBP_delta"),
                "best_epoch": fr.get("p_best_epoch"),
            })
        df = pd.DataFrame(rows)
        self.train_personalized_model.set_df(df)
        show = not df.empty
        self.train_personalized_label.setVisible(show)
        self.train_personalized_table.setVisible(show)
        self.train_personalized_export_btn.setVisible(show)

    def _export_personalized_summary(self):
        df = self.train_personalized_model._df
        if df is None or df.empty:
            QMessageBox.warning(self, "Nothing to export",
                                "No personalized results - run training with 'Personalize' on.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export personalized summary CSV",
            "loso_personalized_summary.csv", "CSV files (*.csv)")
        if path:
            df.to_csv(path, index=False)
            self._train_append_log(f"Exported personalized summary -> {path}")

    def _update_train_scatter(self):
        if not self.fold_predictions:
            return
        all_df = pd.concat(self.fold_predictions.values(), axis=0, ignore_index=True)

        fig = self.train_scatter_canvas.fig
        fig.clear()
        ax1, ax2 = fig.subplots(1, 2)
        for ax, true_col, pred_col, title in [
            (ax1, "SBP_true", "SBP_pred", "SBP: predicted vs true"),
            (ax2, "DBP_true", "DBP_pred", "DBP: predicted vs true"),
        ]:
            ax.scatter(all_df[true_col], all_df[pred_col], s=8, alpha=0.4)
            # Both axes anchored at 0 with a shared [0, hi] range so the y=x line
            # stays a true 45-degree reference.
            hi = max(all_df[true_col].max(), all_df[pred_col].max()) * 1.05
            ax.plot([0, hi], [0, hi], 'r--', linewidth=1, label="y = x")
            ax.set_xlim(0, hi)
            ax.set_ylim(0, hi)
            r = all_df[pred_col].corr(all_df[true_col]) if all_df[true_col].std() > 1e-8 else float('nan')
            ax.text(0.05, 0.95, f"r = {r:.3f}", transform=ax.transAxes,
                    va='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
            ax.set_xlabel("True (mmHg)")
            ax.set_ylabel("Predicted (mmHg)")
            ax.set_title(title)
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.4)
        fig.tight_layout()
        self.train_scatter_canvas.draw()

    def _on_train_all_finished(self, result):
        self.train_run_btn.setEnabled(True)
        self.train_cancel_btn.setEnabled(False)
        self.train_progress_label.setText("Done")
        summary = result["summary"]
        self.train_results_model.set_df(summary)
        self._update_train_scatter()
        mean_sbp = summary["MAE_SBP"].mean()
        mean_dbp = summary["MAE_DBP"].mean()
        self._train_append_log(f"\nAll folds complete. Mean MAE_SBP={mean_sbp:.2f}  Mean MAE_DBP={mean_dbp:.2f}")

        bp_standards = result.get("bp_standards")
        if bp_standards:
            from bp_standards import format_standards_report
            zero_delta_mae = None
            if "MAE_SBP_baseline_calib" in summary and "MAE_DBP_baseline_calib" in summary:
                zero_delta_mae = {"SBP": float(summary["MAE_SBP_baseline_calib"].mean()),
                                  "DBP": float(summary["MAE_DBP_baseline_calib"].mean())}
            self._train_append_log("\n" + format_standards_report(bp_standards, zero_delta_mae))

        # Same device-validation standards, but graded on the PERSONALIZED
        # (fine-tuned) predictions - pooled over each subject's held-out slice.
        # Comparable only to a base pool on the same slices, not the base
        # standards above (whole-subject). Bar shown is the personalized
        # zero-delta baseline.
        bp_standards_personalized = result.get("bp_standards_personalized")
        if bp_standards_personalized:
            from bp_standards import format_standards_report
            p_zero_delta = None
            if "p_baseline_MAE_SBP" in summary and "p_baseline_MAE_DBP" in summary:
                ps = summary.dropna(subset=["p_baseline_MAE_SBP"])
                if len(ps):
                    p_zero_delta = {"SBP": float(ps["p_baseline_MAE_SBP"].mean()),
                                    "DBP": float(ps["p_baseline_MAE_DBP"].mean())}
            self._train_append_log(
                "\n### PERSONALIZED (fine-tuned) model - held-out test slices only ###\n"
                + format_standards_report(bp_standards_personalized, p_zero_delta))

    def _export_training_summary(self):
        df = self.train_results_model._df
        if df is None or df.empty:
            QMessageBox.warning(self, "Nothing to export", "Run LOSO training first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export summary CSV", "loso_summary.csv", "CSV files (*.csv)")
        if path:
            df.to_csv(path, index=False)
            self._train_append_log(f"Exported summary -> {path}")

    def closeEvent(self, event):
        """Stop any running background thread cleanly instead of letting Qt
        tear it down mid-run (which can abort the process)."""
        for worker in (self.worker, self.train_worker, self.test_worker):
            if worker is not None and worker.isRunning():
                worker.request_stop()
                worker.wait(5000)
        event.accept()

def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()