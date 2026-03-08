#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import soundfile as sf

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

def iter_audio_files(root: Path) -> list[Path]:
    return sorted(
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
    )

def get_class_index(class_name: str, class_list: list[str]) -> int:
    return class_list.index(class_name) if class_name in class_list else -1

def write_csv(csv_path: Path, class_index: int, intensity_seq: list[float]) -> None:
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class_index", "intensity"])
        for value in intensity_seq:
            writer.writerow([class_index, round(float(value), 2)])

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate intensity CSVs for audio files.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Input folder with audio files.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output folder for CSV files.")
    parser.add_argument("--fps", type=int, default=75, help="Rows per second of audio.")
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load class list from parameters.json
    param_path = Path(__file__).parent / "raw" / "parameters.json"
    with param_path.open("r") as f:
        params = json.load(f)
    class_list = params["parameter_1"]["classes"]

    audio_files = iter_audio_files(input_dir)
    for audio_path in audio_files:
        class_name = audio_path.parent.name
        class_index = get_class_index(class_name, class_list)
        audio, sr = sf.read(audio_path, dtype="float32")
        duration = len(audio) / sr
        n_rows = max(1, int(round(duration * args.fps)))
        intensity_seq = list(np.linspace(100, 0, n_rows))
        # Copy audio file with class_name_filename format
        audio_base_name = f"{class_name}_{audio_path.stem}{audio_path.suffix}"
        out_audio = output_dir / audio_base_name
        shutil.copy(audio_path, out_audio)
        # Create CSV with matching base name
        csv_base_name = f"{class_name}_{audio_path.stem}.csv"
        out_csv = output_dir / csv_base_name
        write_csv(out_csv, class_index, intensity_seq)
        print(f"Processed {audio_path.name}: {n_rows} rows, intensity 100->0, copied to {out_audio.name}")

if __name__ == "__main__":
    import numpy as np
    main()
