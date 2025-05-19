import os
import glob
import numpy as np
import pandas as pd
import neurokit2 as nk
import mne
import argparse
from tqdm import tqdm


def extract_sleep_stages(annotations, total_seconds, epoch_duration=30):
    stage_events = []
    for annot in annotations:
        desc = annot["description"]
        if "Sleep stage" in desc:
            stage = desc.replace("Sleep stage ", "")
            stage_events.append({
                "onset": annot["onset"],
                "duration": annot["duration"],
                "stage": stage
            })

    df_stage = pd.DataFrame(stage_events)
    if df_stage.empty:
        return None

    num_epochs = int(total_seconds // epoch_duration)
    epoch_df = pd.DataFrame({"epoch": np.arange(num_epochs)})
    epoch_df["start_time"] = epoch_df["epoch"] * epoch_duration
    epoch_df["end_time"] = (epoch_df["epoch"] + 1) * epoch_duration

    df_stage["epoch_start"] = (df_stage["onset"] // epoch_duration).astype(int)

    stage_array = [None] * num_epochs
    for _, row in df_stage.iterrows():
        start = row["epoch_start"]
        end = start + int(np.ceil(row["duration"] / epoch_duration))
        for i in range(start, end):
            if i < num_epochs:
                stage_array[i] = row["stage"]

    # Fill forward
    current = "Unknown"
    for i in range(num_epochs):
        if stage_array[i] is not None:
            current = stage_array[i]
        stage_array[i] = current

    return np.array(stage_array)


def detect_r_peaks(ecg_signal, sfreq):
    _, peaks_dict = nk.ecg_peaks(ecg_signal, sampling_rate=sfreq)
    return peaks_dict["ECG_R_Peaks"]


def compute_heart_rate_2hz(ecg_signal, sfreq, max_hours=8, resample_rate=2):
    r_peaks = detect_r_peaks(ecg_signal, sfreq)
    r_times = r_peaks / sfreq
    ibi = np.diff(r_times)

    if len(ibi) == 0:
        return None, None, None

    # Filter outliers
    mean_ibi = np.mean(ibi)
    std_ibi = np.std(ibi)
    mask = np.abs(ibi - mean_ibi) < 5 * std_ibi
    ibi_filtered = ibi[mask]
    ibi_times = r_times[1:][mask]

    if len(ibi_filtered) == 0:
        return None, None, None

    hr = 1.0 / ibi_filtered
    hr_norm = (hr - np.mean(hr)) / (np.std(hr) + 1e-6)

    t_uniform = np.linspace(0, max_hours * 3600, int(max_hours * 3600 * resample_rate), endpoint=False)
    hr_2hz = np.interp(t_uniform, ibi_times, hr_norm, left=0, right=0)

    return ibi_filtered, ibi_times, hr_2hz


def process_subject(psg_path, scoring_path, output_path):
    try:
        raw = mne.io.read_raw_edf(psg_path, preload=True)
        sfreq = raw.info['sfreq']
        ecg_ch = next((ch for ch in raw.ch_names if "ECG" in ch.upper() or "EKG" in ch.upper()), None)
        if not ecg_ch:
            print(f"No ECG channel found in {psg_path}")
            return None

        raw.filter(0.3, 40, picks=[ecg_ch])
        ecg_signal = raw.get_data(picks=[ecg_ch])[0]

        ibi, ibi_times, hr_2hz = compute_heart_rate_2hz(ecg_signal, sfreq)
        if ibi is None:
            print(f"No valid IBIs for {psg_path}")
            return None

        annotations = mne.read_annotations(scoring_path)
        sleep_stages = extract_sleep_stages(annotations, total_seconds=28800)

        output_data = {
            "ibi": ibi,
            "ibi_times": ibi_times,
            "hr_2hz": hr_2hz
        }
        if sleep_stages is not None:
            output_data["sleep_stages_30s"] = sleep_stages

        filename = os.path.basename(psg_path).replace(".edf", "_ECG_data.npz")
        save_path = os.path.join(output_path, filename)
        np.savez_compressed(save_path, **output_data)
        print(f"Saved to {save_path}")
        return output_data

    except Exception as e:
        print(f"Error processing {psg_path}: {e}")
        return None


def main(args):
    os.makedirs(args.output_path, exist_ok=True)

    psg_files = sorted([f for f in os.listdir(args.input_path) if f.endswith(".edf") and "_sleepscoring" not in f])
    scoring_files = sorted([f for f in os.listdir(args.input_path) if f.endswith("_sleepscoring.edf")])

    assert len(psg_files) == len(scoring_files), "Mismatch in PSG and scoring files count."

    for psg_file, scoring_file in tqdm(zip(psg_files, scoring_files), total=len(psg_files), desc="Processing"):
        process_subject(
            os.path.join(args.input_path, psg_file),
            os.path.join(args.input_path, scoring_file),
            args.output_path
        )

    print("All files processed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ECG Feature Extraction from EDF Files")
    parser.add_argument("--input_path", type=str, required=True, help="Path to input EDF files")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save NPZ output")
    args = parser.parse_args()
    main(args)
