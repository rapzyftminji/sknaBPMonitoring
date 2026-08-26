"""
Recording Cutter
=================
Standalone Qt tool to trim a SKNA/ECG .txt recording and/or its paired
vitals .csv to a chosen time range. Each file gets its own plot with a
draggable shaded region (start/end handles) so you can see exactly what
you're about to cut before saving it.

File conventions matched here (so a cut file stays readable by the rest of
the pipeline in src/):
  - .txt: 14-line BIOPAC export header, data starts at line 15 (skiprows=14),
    4 columns (Time, CH1, CH40, CH41) - same as src/core_processing.py's
    load_txt_signal / src/diagnostics_alignment.py's SKIPROWS=14. The raw
    time column's UNIT is inconsistent across recording batches (some say
    "microSec", newer ones say "milliSec" for the same-shaped column) - see
    src/diagnostics_alignment.py's parse_txt_header. So, like the rest of
    the pipeline, this tool never trusts that column's absolute value:
    it parses the nominal sample interval from the header's "X ms/..." line
    to get fs, then always uses sample_index / fs as the real time axis.
  - .csv: a plain single-header-row vitals export (Date, Time, Systolic
    (mmHg), Diastolic (mmHg), ..., TimeStamp (mS)); elapsed time is derived
    from the TimeStamp (mS) column when present, else row index.

The two panels are intentionally independent (no cross-file alignment) -
each plots its own file on its own native time axis starting at 0. Use the
main app's alignment tools first if you need the two files on one clock
before deciding where to cut.

Run: python3 recording_cutter/main.py
"""
import os
import re

import numpy as np
import pandas as pd
import pyqtgraph as pg
# pyqtgraph's own default is a BLACK plot background - without this, the
# black-pen signal curve below is invisible (black-on-black). Matches
# src/main_window.py's same config call, so this tool's plots look the same.
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton,
    QVBoxLayout, QWidget,
)

TXT_HEADER_LINES = 14        # lines preserved verbatim in any cut .txt output
TXT_SKIPROWS = 14            # matches core_processing.load_txt_signal / diagnostics_alignment.SKIPROWS
TXT_COLUMNS = ("Time_raw", "CH1", "CH40", "CH41")
MAX_PLOT_POINTS = 50_000     # decimate long signals for DISPLAY only; cuts always use full resolution


def _detect_fs(path, default=2000.0):
    """Parse 'X ms/sample' from the header (same convention as
    diagnostics_alignment.parse_txt_header). Falls back to `default` if the
    header doesn't parse - the fs field stays user-editable either way."""
    try:
        with open(path, encoding="latin1") as f:
            head = [f.readline() for _ in range(TXT_HEADER_LINES)]
    except OSError:
        return default
    for line in head:
        m = re.search(r"([0-9.]+)\s*ms", line)
        if m:
            dt_ms = float(m.group(1))
            if dt_ms > 0:
                return 1000.0 / dt_ms
    return default


def _decimate(x, y, max_points=MAX_PLOT_POINTS):
    n = len(x)
    if n <= max_points:
        return x, y
    step = int(np.ceil(n / max_points))
    return x[::step], y[::step]


class CutPanel(QGroupBox):
    """One plot + draggable region + start/end/duration readout + Save
    button. Structurally identical for the raw-signal panel and the vitals
    panel; the owning window decides what "Save" actually writes out.

    Displays time in MINUTES (plain axis label, no `units=` string - that's
    what stops pyqtgraph from auto-switching to SI-prefixed units like "ks"
    on long recordings). save_requested always emits SECONDS regardless of
    the display unit, so callers never have to think about this."""

    save_requested = pyqtSignal(float, float)   # ALWAYS (start_time_sec, end_time_sec)
    SEC_PER_UNIT = 60.0                          # minutes -> seconds

    def __init__(self, title, y_label, parent=None):
        super().__init__(title, parent)
        self.plot = pg.PlotWidget()
        self.plot.setLabel("bottom", "Time (min)")
        self.plot.setLabel("left", y_label)
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.addLegend(offset=(10, 10))

        self.region = pg.LinearRegionItem(brush=pg.mkBrush(255, 200, 0, 60))
        self.region.setZValue(10)
        self.plot.addItem(self.region)
        self.region.sigRegionChanged.connect(self._on_region_changed)

        self.info_label = QLabel("Load a file to begin.")
        self.save_btn = QPushButton("Save cut...")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self._on_save_clicked)

        layout = QVBoxLayout(self)
        layout.addWidget(self.plot)
        row = QHBoxLayout()
        row.addWidget(self.info_label, stretch=1)
        row.addWidget(self.save_btn)
        layout.addLayout(row)

        self._data_max_time = 0.0

    def set_curve(self, x_sec, y, pen, x_full_max_sec, name=None):
        """(Re)plot the panel's primary curve and reset the region to span
        the whole file. Takes SECONDS in (x_sec, x_full_max_sec) and converts
        to minutes internally for display, so callers never convert units
        themselves. Call add_curve() afterward for extra overlays (e.g. DBP
        alongside SBP) - it does NOT reset the region."""
        self.plot.clear()
        self.plot.addItem(self.region)
        self.plot.plot(np.asarray(x_sec) / self.SEC_PER_UNIT, y, pen=pen, name=name)
        self._data_max_time = max(x_full_max_sec / self.SEC_PER_UNIT, 1e-6)
        self.region.setBounds([0, self._data_max_time])
        self.region.setRegion([0, self._data_max_time])
        self.save_btn.setEnabled(True)
        self._on_region_changed()

    def add_curve(self, x_sec, y, pen, name=None):
        self.plot.plot(np.asarray(x_sec) / self.SEC_PER_UNIT, y, pen=pen, name=name)

    def get_region_seconds(self):
        lo, hi = self.region.getRegion()
        return lo * self.SEC_PER_UNIT, hi * self.SEC_PER_UNIT

    def _on_region_changed(self):
        lo, hi = self.region.getRegion()   # minutes
        self.info_label.setText(
            f"Start: {lo:.3f} min   End: {hi:.3f} min   Duration: {hi - lo:.3f} min "
            f"(of {self._data_max_time:.2f} min total)")

    def _on_save_clicked(self):
        start_s, end_s = self.get_region_seconds()
        self.save_requested.emit(start_s, end_s)


class RecordingCutterWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Recording Cutter")
        self.resize(1200, 850)

        self.txt_path = None
        self.csv_path = None
        self._txt_time = None      # seconds, full resolution
        self._txt_ch1 = None
        self._txt_fs = None
        self._csv_df = None
        self._csv_time = None      # seconds elapsed, full resolution

        # ---- file pickers ----
        self.txt_edit = QLineEdit()
        self.txt_edit.setReadOnly(True)
        txt_browse = QPushButton("Browse...")
        txt_browse.clicked.connect(self._browse_txt)

        self.csv_edit = QLineEdit()
        self.csv_edit.setReadOnly(True)
        csv_browse = QPushButton("Browse...")
        csv_browse.clicked.connect(self._browse_csv)

        self.fs_spin = QDoubleSpinBox()
        self.fs_spin.setRange(1, 200000)
        self.fs_spin.setDecimals(1)
        self.fs_spin.setValue(2000.0)
        self.fs_spin.setSuffix(" Hz")
        self.fs_spin.setToolTip(
            "Auto-detected from the .txt header's 'X ms/sample' line. Override "
            "if it looks wrong - the raw time column's unit is not consistent "
            "across recording batches, so this value (not that column) sets "
            "the plotted time axis.")
        self.fs_spin.valueChanged.connect(self._on_fs_changed)

        form = QFormLayout()
        row1 = QHBoxLayout()
        row1.addWidget(self.txt_edit)
        row1.addWidget(txt_browse)
        form.addRow("SKNA/ECG .txt:", row1)
        row2 = QHBoxLayout()
        row2.addWidget(self.csv_edit)
        row2.addWidget(csv_browse)
        form.addRow("Vitals .csv:", row2)
        form.addRow("Sampling rate:", self.fs_spin)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)

        self.txt_panel = CutPanel("SKNA/ECG signal (CH1)", "Amplitude")
        self.txt_panel.save_requested.connect(self._save_txt_cut)
        self.csv_panel = CutPanel("Vitals (SBP / DBP)", "mmHg")
        self.csv_panel.save_requested.connect(self._save_csv_cut)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.status_label)
        layout.addWidget(self.txt_panel, stretch=1)
        layout.addWidget(self.csv_panel, stretch=1)

    # ----------------------------------------------------------------
    # txt
    # ----------------------------------------------------------------
    def _browse_txt(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select SKNA/ECG .txt", "", "Text files (*.txt);;All files (*)")
        if path:
            self._load_txt(path)

    def _load_txt(self, path):
        fs = _detect_fs(path)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            # Only Time + CH1 are loaded (not CH40/CH41) to keep memory bounded
            # on the largest recordings (~900 MB / ~19M rows at 10 kHz) - the
            # cut operation re-reads the full 4 columns later, but only for
            # the exact row range being kept, never the whole file at once.
            df = pd.read_csv(
                path, skiprows=TXT_SKIPROWS, usecols=(0, 1), names=["Time_raw", "CH1"],
                dtype={"Time_raw": np.float64, "CH1": np.float32}, encoding="latin1",
            )
        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Load failed", f"Could not read {path}:\n{e}")
            return
        QApplication.restoreOverrideCursor()

        self.txt_path = path
        self.txt_edit.setText(path)
        self._txt_fs = fs
        self.fs_spin.blockSignals(True)
        self.fs_spin.setValue(fs)
        self.fs_spin.blockSignals(False)
        n = len(df)
        self._txt_time = np.arange(n) / fs
        self._txt_ch1 = df["CH1"].to_numpy()
        self._replot_txt()
        self._append_status(
            f"Loaded {n:,} samples @ {fs:.0f} Hz from {os.path.basename(path)} "
            f"({self._txt_time[-1]:.1f}s)")

    def _on_fs_changed(self, value):
        if self._txt_ch1 is None or value <= 0:
            return
        self._txt_fs = value
        self._txt_time = np.arange(len(self._txt_ch1)) / value
        self._replot_txt()

    def _replot_txt(self):
        x_plot, y_plot = _decimate(self._txt_time, self._txt_ch1)
        self.txt_panel.set_curve(x_plot, y_plot, pen=pg.mkPen('k'),
                                  x_full_max_sec=self._txt_time[-1])

    def _save_txt_cut(self, start_s, end_s):
        if self.txt_path is None or self._txt_time is None:
            return
        start_idx = int(np.searchsorted(self._txt_time, start_s, side="left"))
        end_idx = int(np.searchsorted(self._txt_time, end_s, side="right"))
        start_idx = max(0, min(start_idx, len(self._txt_time) - 1))
        end_idx = max(start_idx + 1, min(end_idx, len(self._txt_time)))
        n_rows = end_idx - start_idx

        default_path = self._default_cut_name(self.txt_path)
        out_path, _ = QFileDialog.getSaveFileName(
            self, "Save cut .txt", default_path, "Text files (*.txt)")
        if not out_path:
            return

        self._append_status(f"Saving {n_rows:,} rows to {os.path.basename(out_path)}... "
                             f"(re-reads the source file for just this row range, may take "
                             f"a moment on large recordings)")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        try:
            with open(self.txt_path, encoding="latin1") as f:
                header_lines = [f.readline() for _ in range(TXT_HEADER_LINES)]
            # Full 4 columns, but ONLY for the rows being kept - the source
            # file's full column set is never loaded into memory at once.
            chunk = pd.read_csv(
                self.txt_path, skiprows=TXT_SKIPROWS + start_idx, nrows=n_rows,
                usecols=(0, 1, 2, 3), names=list(TXT_COLUMNS), header=None,
                encoding="latin1",
            )
            with open(out_path, "w", encoding="latin1", newline="") as f:
                f.writelines(header_lines)
                chunk.to_csv(f, header=False, index=False, lineterminator="\n")
        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Save failed", f"Could not write {out_path}:\n{e}")
            return
        QApplication.restoreOverrideCursor()
        self._append_status(f"Saved {n_rows:,} rows ({end_s - start_s:.1f}s) -> {out_path}")
        QMessageBox.information(self, "Saved",
                                 f"Wrote {n_rows:,} rows ({end_s - start_s:.1f}s) to:\n{out_path}")

    # ----------------------------------------------------------------
    # csv
    # ----------------------------------------------------------------
    def _browse_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select vitals .csv", "", "CSV files (*.csv);;All files (*)")
        if path:
            self._load_csv(path)

    def _load_csv(self, path):
        try:
            df = pd.read_csv(path)
        except Exception as e:
            QMessageBox.critical(self, "Load failed", f"Could not read {path}:\n{e}")
            return

        ts_col = next((c for c in df.columns if "TimeStamp" in c), None)
        if ts_col is not None:
            t = (df[ts_col] - df[ts_col].iloc[0]).to_numpy(dtype=float) / 1000.0
        else:
            t = df.index.to_numpy(dtype=float)

        self.csv_path = path
        self.csv_edit.setText(path)
        self._csv_df = df
        self._csv_time = t

        sbp_col = next((c for c in df.columns if "Systolic" in c), None)
        dbp_col = next((c for c in df.columns if "Diastolic" in c), None)
        x_max = float(t[-1]) if len(t) else 0.0
        if sbp_col is not None:
            self.csv_panel.set_curve(t, df[sbp_col].to_numpy(), pen=pg.mkPen('b', width=2),
                                      x_full_max_sec=x_max, name="SBP")
        else:
            self.csv_panel.set_curve(t, np.zeros(len(df)), pen=pg.mkPen('b', width=2), x_full_max_sec=x_max)
        if dbp_col is not None:
            self.csv_panel.add_curve(t, df[dbp_col].to_numpy(), pen=pg.mkPen('r', width=2), name="DBP")

        self._append_status(f"Loaded {len(df):,} vitals rows from {os.path.basename(path)} "
                             f"({x_max:.1f}s)")

    def _save_csv_cut(self, start_s, end_s):
        if self.csv_path is None or self._csv_time is None:
            return
        mask = (self._csv_time >= start_s) & (self._csv_time <= end_s)
        n_rows = int(mask.sum())
        if n_rows == 0:
            QMessageBox.warning(self, "Nothing to save",
                                 "No vitals rows fall inside the selected range.")
            return
        default_path = self._default_cut_name(self.csv_path)
        out_path, _ = QFileDialog.getSaveFileName(
            self, "Save cut .csv", default_path, "CSV files (*.csv)")
        if not out_path:
            return
        try:
            self._csv_df.loc[mask].to_csv(out_path, index=False)
        except Exception as e:
            QMessageBox.critical(self, "Save failed", f"Could not write {out_path}:\n{e}")
            return
        self._append_status(f"Saved {n_rows:,} rows ({end_s - start_s:.1f}s) -> {out_path}")
        QMessageBox.information(self, "Saved",
                                 f"Wrote {n_rows:,} rows ({end_s - start_s:.1f}s) to:\n{out_path}")

    # ----------------------------------------------------------------
    def _append_status(self, msg):
        self.status_label.setText(msg)

    @staticmethod
    def _default_cut_name(path):
        base, ext = os.path.splitext(path)
        return f"{base}_cut{ext}"
