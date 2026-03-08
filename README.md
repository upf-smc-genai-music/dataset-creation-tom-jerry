# Dataset Creation: Tom & Jerry

This repository provides scripts and data for generating and augmenting audio datasets.

## Structure
- `raw/`: CSV files and parameters for dataset generation.
- `scripts/`: Python scripts for data processing and augmentation.

## Setup
1. **Clone the repo**
2. **Create Conda environment**:
   ```bash
   conda create -n genai_ass2 python=3.10 -y
   conda activate genai_ass2
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
  ```bash
  python intensity_csv_generator.py
  ```
  ```bash
  python augment_transpose.py
  ```
  ```bash
  python lowpass_sweep.py
  ```
### Script Explanations

- **intensity_csv_generator.py**: Scans audio files, generates CSVs with intensity values (from 100 to 0) for each file, and copies audio files with standardized names.
- **augment_transpose.py**: Augments the dataset by pitch-shifting each audio file by -1 and +1 semitone, creating new versions and optionally copying originals.
- **lowpass_sweep.py**: Applies a time-varying low-pass filter to each audio file, with cutoff frequency sweeping from a start to end value (set in parameters.json). Outputs filtered audio and corresponding CSVs for each file.

## Notes
- Place your audio files in the appropriate folders before running scripts.
- Adjust parameters in `raw/parameters.json` as needed.