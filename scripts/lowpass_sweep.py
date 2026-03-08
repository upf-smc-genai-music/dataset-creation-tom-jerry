#!/usr/bin/env python3
from __future__ import annotations


import argparse
import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import scipy.signal

SUPPORTED_AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".flac",
    ".aac",
    ".ogg",
    ".m4a",
    ".wma",
    ".aiff",
    ".au",
    ".ra",
    ".3gp",
    ".amr",
    ".ac3",
    ".dts",
    ".ape",
    ".mka",
    ".opus",
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply a low-pass filter with linearly varying cutoff frequency to each audio file in a folder."
    )
    parser.add_argument("--input-dir", type=Path, required=True, help="Input folder with audio files.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output folder for filtered audio and CSV.")
    # Remove start-cutoff and end-cutoff arguments, will be read from parameters.json
    parser.add_argument("--fps", type=int, default=75, help="Frames per second for CSV annotation.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output files if they exist.")
    return parser.parse_args()

def iter_audio_files(root: Path) -> list[Path]:
    return sorted(
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
    )

def apply_time_varying_lowpass(audio: np.ndarray, sr: int, cutoff_seq: np.ndarray) -> np.ndarray:
    # Apply a low-pass filter with cutoff changing per frame
    filtered = np.zeros_like(audio)
    n_frames = len(cutoff_seq)
    frame_len = int(sr / 75)
    for i in range(n_frames):
        start = i * frame_len
        end = min((i + 1) * frame_len, len(audio))
        if end <= start:
            continue
        segment = audio[start:end]
        # If segment is too short for filtering, just copy original
        if len(segment) < 16:
            filtered[start:end] = segment
            continue
        cutoff = cutoff_seq[i]
        nyq = sr / 2
        norm_cutoff = min(cutoff / nyq, 0.99)
        b, a = scipy.signal.butter(4, norm_cutoff, btype="low")
        try:
            filtered[start:end] = scipy.signal.filtfilt(b, a, segment)
        except ValueError:
            filtered[start:end] = segment
    return filtered

def write_csv(csv_path: Path, cutoff_seq: np.ndarray) -> None:
    print(csv_path)
    class_name = csv_path.stem.split('-')[0]
    class_name = class_name[:-2]
    print(f"Class name: {class_name}")
    # Load class list from parameters.json
    param_path = Path(__file__).parent / "raw" / "parameters.json"
    with param_path.open("r") as f:
        params = json.load(f)
    class_list = params["parameter_1"]["classes"]
    class_index = class_list.index(class_name) if class_name in class_list else -1
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class_index", "cutoff_freq_Hz"])
        for value in cutoff_seq:
            writer.writerow([class_index, round(float(value), 2)])

def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load cutoff parameters from parameters.json
    param_path = Path(__file__).parent / "raw" / "parameters.json"
    with param_path.open("r") as f:
        params = json.load(f)
    start_cutoff = params["parameter_2"]["start_cutoff"]
    end_cutoff = params["parameter_2"]["end_cutoff"]

    audio_files = iter_audio_files(input_dir)
    for audio_path in audio_files:
        class_name = audio_path.parent.name
        base_name = f"{class_name}_{audio_path.stem}{audio_path.suffix}"
        out_audio = output_dir / base_name
        out_csv = out_audio.with_suffix(".csv")
        if not args.overwrite and out_audio.exists():
            print(f"Skipping {out_audio.name} (already exists)")
            continue

        audio, sr = sf.read(audio_path, dtype="float32")
        duration = len(audio) / sr
        n_frames = max(1, int(round(duration * args.fps)))
        cutoff_seq = np.linspace(start_cutoff, end_cutoff, n_frames)

        filtered = apply_time_varying_lowpass(audio, sr, cutoff_seq)
        sf.write(out_audio, filtered, sr)
        write_csv(out_csv, cutoff_seq)
        print(f"Processed {out_audio.name}: {n_frames} frames, cutoff {start_cutoff}->{end_cutoff} Hz")

if __name__ == "__main__":
    main()