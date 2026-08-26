#!/usr/bin/env python3
"""Upsample BIOPAC AcqKnowledge text exports to a common sampling rate.

The newer recordings (Jose, Rizmi, Nhu) were acquired at 10 kHz (0.1 ms/sample)
while every earlier dataset (alice, arthur, becky, cindy, cindy2, eric) is at
2 kHz (0.5 ms/sample). Because the analysis pipeline takes the sampling rate
from a manual spin-box (it is NOT read from the file header), mixing rates
silently corrupts every frequency-domain feature. This tool resamples the slow
recordings up to the common 10 kHz so all datasets share one rate.

Interpolation is done with scipy.signal.resample_poly, a polyphase resampler
whose zero-phase (linear-phase, group-delay-compensated) FIR anti-imaging filter
cuts off at the *original* Nyquist (1 kHz for a 5x upsample). No genuine signal
lives above the source Nyquist, so the full 0-1000 Hz band -- including the
500-1000 Hz SKNA band -- passes through untouched while the spectral images the
zero-stuffing would create above 1 kHz are removed. Edges are padded with a
linear fit ('line') so the large DC offset on the BP channel does not ring.

The output is a byte-compatible drop-in for the existing loader
(core_processing.load_txt_signal / worker.py): 13 header lines followed by
`Time_ms,CH1,CH40,CH41,` rows with a trailing comma. Only the sample-interval
and sample-count header lines are patched to reflect the new rate.

Usage:
    python src/upsample_txt.py dataset/txt/SKNA_BP_alice.txt \
                               dataset/txt/SKNA_BP_arthur.txt \
                               dataset/txt/SKNA_BP_becky.txt \
                               dataset/txt/SKNA_BP_cindy.txt \
                               dataset/txt/SKNA_BP_cindy2.txt \
                               dataset/txt/SKNA_BP_eric.txt
    # -> writes *_10kHz.txt next to each input

    python src/upsample_txt.py INPUT.txt --target-fs 10000 --out-dir dataset/txt
    python src/upsample_txt.py INPUT.txt --factor 5          # force the factor
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

N_HEADER_LINES = 13          # lines before the first data row
INTERVAL_LINE = 3            # 0-indexed header line holding "<x> ms/sample"
SAMPLES_LINE = 12            # 0-indexed header line holding "<N> samples,..."
CHANNEL_COLS = (1, 2, 3)     # CH1, CH40, CH41 (col 0 is time, regenerated)
CHANNEL_FMT = "%.8g"
TIME_FMT = "%.10g"


def read_header(path):
    """Return the first N_HEADER_LINES raw (latin-1 decoded) header lines."""
    lines = []
    with open(path, "rb") as f:
        for _ in range(N_HEADER_LINES):
            lines.append(f.readline().decode("latin-1").rstrip("\r\n"))
    return lines


def parse_interval_ms(header):
    """Parse the per-sample interval in milliseconds from the header.

    The interval line looks like '0.5 ms/sample' (with trailing binary junk).
    A 'microSec' time-column label does not change this line, so it is a
    reliable source for the true interval.
    """
    m = re.search(r"([-+]?\d*\.?\d+)\s*ms", header[INTERVAL_LINE])
    if not m:
        raise ValueError(
            f"could not parse sample interval from header line: "
            f"{header[INTERVAL_LINE]!r}"
        )
    return float(m.group(1))


def patch_header(header, n_out, target_interval_ms):
    """Return header lines with interval and sample-count updated for the new rate."""
    out = list(header)
    # interval line: replace the leading numeric value, keep the rest verbatim
    out[INTERVAL_LINE] = re.sub(
        r"([-+]?\d*\.?\d+)(\s*ms)",
        lambda m: f"{target_interval_ms:g}{m.group(2)}",
        out[INTERVAL_LINE],
        count=1,
    )
    # sample-count line: ",N samples,N samples,N samples,"
    out[SAMPLES_LINE] = f",{n_out} samples,{n_out} samples,{n_out} samples,"
    return out


def upsample_file(in_path, out_path, target_fs, factor=None, chunk_log=True):
    t0 = time.time()
    header = read_header(in_path)
    src_interval_ms = parse_interval_ms(header)
    src_fs = 1000.0 / src_interval_ms
    target_interval_ms = 1000.0 / target_fs

    if factor is None:
        ratio = target_fs / src_fs
        factor = int(round(ratio))
        if factor < 1 or abs(ratio - factor) > 1e-6:
            raise ValueError(
                f"{os.path.basename(in_path)}: target {target_fs:g} Hz is not an "
                f"integer multiple of source {src_fs:g} Hz (ratio={ratio:.4f}). "
                f"Pass --factor to override."
            )

    print(
        f"[{os.path.basename(in_path)}] source={src_fs:g} Hz "
        f"({src_interval_ms:g} ms) -> target={target_fs:g} Hz "
        f"({target_interval_ms:g} ms), factor={factor}"
    )
    if factor == 1:
        print("  factor is 1 (already at target rate) - nothing to do; skipping.")
        return

    # Read the three signal channels (time column is regenerated, not read).
    print("  reading data ...")
    df = pd.read_csv(
        in_path,
        skiprows=N_HEADER_LINES,
        header=None,
        usecols=list(CHANNEL_COLS),
        dtype=np.float32,
        encoding="latin1",
        engine="c",
        na_filter=False,
    )
    data = df.to_numpy()
    n_in = data.shape[0]
    print(f"  read {n_in:,} samples x {data.shape[1]} channels "
          f"({time.time() - t0:.1f}s)")

    # Upsample each channel with a zero-phase polyphase FIR anti-imaging filter.
    # padtype='line' extends the edges with a linear fit so the BP channel's
    # large DC offset does not produce edge ringing.
    print(f"  upsampling by {factor} (polyphase FIR anti-image @ {src_fs/2:g} Hz)...")
    up_cols = [
        resample_poly(data[:, c].astype(np.float64), factor, 1, padtype="line")
        for c in range(data.shape[1])
    ]
    n_out = min(len(c) for c in up_cols)
    up = np.column_stack([c[:n_out] for c in up_cols])
    print(f"  upsampled to {n_out:,} samples")

    # Regenerate the time column at the new interval (in the same ms units).
    t_ms = np.arange(n_out, dtype=np.float64) * target_interval_ms
    out_arr = np.column_stack([t_ms, up])

    new_header = patch_header(header, n_out, target_interval_ms)

    # Write header (latin-1) then data rows with a trailing comma to match format.
    fmt = TIME_FMT + "," + ",".join([CHANNEL_FMT] * up.shape[1]) + ","
    print(f"  writing {out_path} ...")
    with open(out_path, "w", encoding="latin-1", newline="\n") as f:
        f.write("\n".join(new_header) + "\n")
        np.savetxt(f, out_arr, fmt=fmt)

    print(f"  done: {out_path}  ({time.time() - t0:.1f}s total)\n")


def default_out_path(in_path, out_dir, target_fs):
    base, ext = os.path.splitext(os.path.basename(in_path))
    khz = target_fs / 1000.0
    tag = f"_{khz:g}kHz" if khz == int(khz) else f"_{target_fs:g}Hz"
    out_name = f"{base}{tag}{ext}"
    return os.path.join(out_dir or os.path.dirname(in_path), out_name)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="BIOPAC .txt export(s) to upsample")
    ap.add_argument("--target-fs", type=float, default=10000.0,
                    help="target sampling rate in Hz (default: 10000)")
    ap.add_argument("--factor", type=int, default=None,
                    help="force integer upsampling factor (default: auto from header)")
    ap.add_argument("--out-dir", default=None,
                    help="output directory (default: alongside each input)")
    ap.add_argument("--overwrite", action="store_true",
                    help="overwrite existing output files")
    args = ap.parse_args(argv)

    for in_path in args.inputs:
        if not os.path.isfile(in_path):
            print(f"!! not found: {in_path}", file=sys.stderr)
            continue
        out_path = default_out_path(in_path, args.out_dir, args.target_fs)
        if os.path.abspath(out_path) == os.path.abspath(in_path):
            print(f"!! refusing to overwrite input: {in_path}", file=sys.stderr)
            continue
        if os.path.exists(out_path) and not args.overwrite:
            print(f"!! exists (use --overwrite): {out_path}", file=sys.stderr)
            continue
        upsample_file(in_path, out_path, args.target_fs, factor=args.factor)


if __name__ == "__main__":
    main()
