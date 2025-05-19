# Sleep Stage Classification from ECG/PPG-Derived Heart Rate

This repository contains a complete pipeline for sleep stage classification using heart rate (HR) signals derived from ECG and PPG. It includes:

- Preprocessing of raw sleep study `.edf` files
- HR feature extraction (IHR at 2Hz and IBI)
- Model training using patch-based CNNs and dilated convolutions
- Fine-tuning on PPG-based HR signals

---

##Installation

Install all required packages using:

```bash
pip install numpy pandas scikit-learn tensorflow neurokit2 mne tqdm
```

---

##  ECG Preprocessing

Use this step to convert `.edf` PSG and scoring files into `.npz` files containing the following keys:

- `hr_2hz`: 2Hz instantaneous heart rate
- `ibi`: inter-beat intervals
- `ibi_times`: time points of IBI events
- `sleep_stages_30s`: one label per 30-second epoch (8 hours → 1200 epochs)

Run the preprocessing script:

```bash
python preprocessing/ecg_preprocessing.py \
    --input_path /path/to/edf_files \
    --output_path /path/to/output_npz
```

Ensure each PSG `.edf` has a matching `_sleepscoring.edf` annotation file.

---

##  Step 2: ECG Model Training

Train a sleep stage classifier using the `.npz` files generated from ECG data.

```bash
python training/ecg_training.py \
    --base_path ./data \
    --model_path ./models/weights \
    --checkpoint_path ./models/best_model.h5
```

### Label Mapping (3 classes):

- Wake (W): `0`  
- NREM (N1, N2, N3): `1`  
- REM (R): `2`

---

## Step 3: Fine-Tune on PPG derived heart-rate data

Fine-tune the pretrained ECG model using PPG-derived `.npz` files.

```bash
python training/ppg_training.py \
    --input_dir ./data \
    --model_weights ./models/best_model.h5 \
    --checkpoint_path ./models/ppg_checkpoint.h5 \
    --output_dir ./models/ppg_finetuned \
    --epochs 1000 \
    --batch_size 2
```

---
