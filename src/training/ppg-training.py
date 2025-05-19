import os
import glob
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from models.heart_rate_model import build_full_model 
import tensorflow as tf


def load_ppg_data(npz_files):
    X, y = [], []
    for f in npz_files:
        data = np.load(f)
        if "hr_2hz" not in data or "sleep_stages_30s" not in data:
            continue
        X.append(data["hr_2hz"])
        y.append(data["sleep_stages_30s"])
    return np.stack(X), np.stack(y)


def setup_callbacks(save_path="best_model.h5"):
    checkpoint = ModelCheckpoint(save_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
    early_stop = EarlyStopping(monitor='val_accuracy', patience=20, mode='max', verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', patience=5, mode='max', verbose=1)
    return [checkpoint, early_stop, reduce_lr]


def main(args):
    files = sorted(glob.glob(os.path.join(args.input_dir, "*.npz")))
    print(f"Total NPZ files found: {len(files)}")

    train_ids, test_ids = train_test_split(files, test_size=0.15, random_state=1338)
    train_files, val_files = train_test_split(train_ids, test_size=0.1, random_state=1337)

    print(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_ids)}")

    X_train, y_train = load_ppg_data(train_files)
    X_val, y_val = load_ppg_data(val_files)
    X_test, y_test = load_ppg_data(test_ids)

    print("Input shape:", X_train.shape)
    print("Label shape:", y_train.shape)


    # Build and load pretrained model
    model = build_full_model(
        input_length=X_train.shape[1],
        patch_length=256,
        patch_step=59,
        num_patches=960,
        num_classes=3,   # Assuming 3-class sleep staging
        dropout_rate=0.2
    )
    model.load_weights(args.model_weights)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # Train
    callbacks = setup_callbacks(args.checkpoint_path)
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks
    )

    # Save final weights
    os.makedirs(args.output_dir, exist_ok=True)
    final_path = os.path.join(args.output_dir, "ppg_finetuned_model.h5")
    model.save_weights(final_path)
    print(f"Final model saved to: {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune ECG model for PPG-based sleep staging")
    parser.add_argument('--input_dir', type=str, required=True, help="Directory containing .npz PPG files")
    parser.add_argument('--model_weights', type=str, required=True, help="Path to pre-trained ECG model weights")
    parser.add_argument('--checkpoint_path', type=str, default="best_model.h5", help="Path to save best model")
    parser.add_argument('--output_dir', type=str, default=".", help="Where to save final model")
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=2)
    args = parser.parse_args()
    main(args)
