import os
import sys
import glob
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# parent directory to path to import from models/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.heart_rate_model import build_full_model


def load_data(files, mapping):
    X_list, y_list = [], []
    for f in files:
        data = np.load(f)
        hr_2hz = data["hr_2hz"]             # shape: (72000,)
        stages_30s = np.array([mapping[stage] for stage in data["sleep_stages_30s"]])  # shape: (1200,)
        X_list.append(hr_2hz)
        y_list.append(stages_30s)
    return np.stack(X_list), np.stack(y_list)


def parse_args():
    parser = argparse.ArgumentParser(description="Train heart rate model for sleep stage classification")
    parser.add_argument('--base_path', type=str, required=True, help='Path to the folder containing .npz files')
    parser.add_argument('--model_path', type=str, required=True, help='Directory to save trained model weights')
    parser.add_argument('--checkpoint_path', type=str, default='best_model.h5', help='Path to save best model checkpoint')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs')
    return parser.parse_args()


def main():
    args = parse_args()

    mapping = {'W': 0, 'N1': 1, 'N2': 1, 'N3': 1, 'R': 2}
    all_files = sorted(glob.glob(os.path.join(args.base_path, "*.npz")))

    # Train/val/test split
    train_ids, test_ids = train_test_split(all_files, test_size=0.15, random_state=1338)
    train_files, val_files = train_test_split(train_ids, test_size=0.1, random_state=1337)

    X_train, y_train = load_data(train_files, mapping)
    X_val, y_val = load_data(val_files, mapping)
    X_test, y_test = load_data(test_ids, mapping)

    print("Train shape:", X_train.shape, y_train.shape)
    print("Val shape:", X_val.shape, y_val.shape)
    print("Test shape:", X_test.shape, y_test.shape)

    # Build and compile model
    model = build_full_model(
        input_length=57600,
        patch_length=256,
        patch_step=59,
        num_patches=960,
        num_classes=3,
        dropout_rate=0.2
    )
    model.compile(optimizer=Adam(), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    # Define callbacks
    callbacks = [
        ModelCheckpoint(args.checkpoint_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1),
        EarlyStopping(monitor='val_accuracy', patience=20, mode='max', verbose=1),
        ReduceLROnPlateau(monitor='val_accuracy', patience=5, mode='max', verbose=1)
    ]

    # Train the model
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks
    )

    # Save final weights
    os.makedirs(args.model_path, exist_ok=True)
    model.save_weights(os.path.join(args.model_path, "ecg-hrv-model.weights.h5"))
    print(f"Model weights saved to {os.path.join(args.model_path, 'ecg-hrv-model.weights.h5')}")


if __name__ == "__main__":
    main()
