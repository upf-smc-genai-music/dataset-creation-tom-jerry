#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import librosa
import numpy as np
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Augment dataset by transposing each audio file by -1 and +1 semitone."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("raw_audios"),
        help="Root folder containing source audio files (subfolders preserved).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("raw_audios_augmented"),
        help="Root folder where augmented audio files are written.",
    )
    parser.add_argument(
        "--copy-originals",
        action="store_true",
        help="Also copy original files to output-dir alongside augmented versions.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output files if they already exist.",
    )
    return parser.parse_args()


def iter_audio_files(root: Path) -> list[Path]:
    return sorted(
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
    )


def pitch_shift_audio(audio: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    if audio.ndim == 1:
        shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=semitones)
        return shifted.astype(np.float32)

    shifted_channels = []
    for channel_idx in range(audio.shape[1]):
        shifted_channel = librosa.effects.pitch_shift(
            audio[:, channel_idx], sr=sr, n_steps=semitones
        )
        shifted_channels.append(shifted_channel)
    shifted_stacked = np.stack(shifted_channels, axis=1)
    return shifted_stacked.astype(np.float32)


def write_augmented_pair(
    source_audio: Path,
    destination_base: Path,
    sr: int,
    audio: np.ndarray,
    semitone_shift: int,
    suffix: str,
    overwrite: bool,
) -> bool:
    destination_audio = destination_base.with_name(f"{destination_base.name}-{suffix}").with_suffix(
        ".wav"
    )
    destination_csv = destination_audio.with_suffix(".csv")

    if destination_audio.exists() and not overwrite:
        return False

    shifted = pitch_shift_audio(audio, sr=sr, semitones=semitone_shift)
    sf.write(destination_audio, shifted, sr)

    source_csv = source_audio.with_suffix(".csv")
    if source_csv.exists():
        shutil.copy2(source_csv, destination_csv)

    return True


def copy_original_files(source_audio: Path, destination_audio: Path, overwrite: bool) -> None:
    destination_audio.parent.mkdir(parents=True, exist_ok=True)
    if overwrite or not destination_audio.exists():
        shutil.copy2(source_audio, destination_audio)

    source_csv = source_audio.with_suffix(".csv")
    destination_csv = destination_audio.with_suffix(".csv")
    if source_csv.exists() and (overwrite or not destination_csv.exists()):
        shutil.copy2(source_csv, destination_csv)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    audio_files = iter_audio_files(input_dir)
    if not audio_files:
        raise RuntimeError(f"No supported audio files found under: {input_dir}")

    created = 0
    skipped = 0

    for audio_path in audio_files:
        relative = audio_path.relative_to(input_dir)
        destination_original = output_dir / relative
        destination_original.parent.mkdir(parents=True, exist_ok=True)

        audio, sr = sf.read(audio_path, dtype="float32", always_2d=False)
        destination_base = destination_original.with_suffix("")

        wrote_down = write_augmented_pair(
            source_audio=audio_path,
            destination_base=destination_base,
            sr=sr,
            audio=audio,
            semitone_shift=-1,
            suffix="down1st",
            overwrite=args.overwrite,
        )
        wrote_up = write_augmented_pair(
            source_audio=audio_path,
            destination_base=destination_base,
            sr=sr,
            audio=audio,
            semitone_shift=1,
            suffix="up1st",
            overwrite=args.overwrite,
        )

        if args.copy_originals:
            copy_original_files(audio_path, destination_original, args.overwrite)

        if wrote_down:
            created += 1
        else:
            skipped += 1

        if wrote_up:
            created += 1
        else:
            skipped += 1

    print(f"Processed source files: {len(audio_files)}")
    print(f"Augmented files created: {created}")
    print(f"Augmented files skipped: {skipped}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()