import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# -------------------------
# 1. Create a custom layer to handle framing
# -------------------------
class FrameLayer(layers.Layer):
    """
    A layer that wraps tf.signal.frame(...) to handle a symbolic Keras tensor.
    """
    def __init__(self, frame_length, frame_step, num_patches=None, **kwargs):
        super().__init__(**kwargs)
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.num_patches = num_patches

    def call(self, inputs, **kwargs):
        # inputs shape: (batch_size, input_length)
        # Perform framing
        x = tf.signal.frame(inputs, frame_length=self.frame_length, frame_step=self.frame_step)
        # If num_patches is specified, slice that many frames
        if self.num_patches is not None:
            x = x[:, :self.num_patches, :]
        return x

# -------------------------
# 2. Convolutional blocks for local feature extraction
# -------------------------
def conv_block(x, filters, kernel_size=3, dropout_rate=0.2):
    """
    A single convolutional block:
      - Two Conv1D layers (kernel_size=3, dilation_rate=1, LeakyReLU activation)
      - MaxPool1D(pool_size=2) on both the main path and the shortcut
      - Residual connection from a downsampled version of the input
    """
    # Shortcut is a downsample of x
    shortcut = layers.MaxPooling1D(pool_size=2)(x)

    # First convolution
    x = layers.Conv1D(filters, kernel_size, padding="same", dilation_rate=1)(x)
    x = layers.LeakyReLU(alpha=0.01)(x)

    # Second convolution
    x = layers.Conv1D(filters, kernel_size, padding="same", dilation_rate=1)(x)
    x = layers.LeakyReLU(alpha=0.01)(x)

    # Downsample x
    x = layers.MaxPooling1D(pool_size=2)(x)

    # Residual
    x = layers.Add()([x, shortcut])
    return x

def build_local_extractor(input_shape=(256, 1)):
    """
    Builds a CNN for local feature extraction from each patch of shape (256,1).
    Outputs a 128-dimensional embedding.
    """
    inputs = keras.Input(shape=input_shape)
    x = inputs

    # Three convolutional blocks
    x = conv_block(x, filters=128, kernel_size=3, dropout_rate=0.2)
    x = conv_block(x, filters=128, kernel_size=3, dropout_rate=0.2)
    x = conv_block(x, filters=128, kernel_size=3, dropout_rate=0.2)

    # Global average + Dense => 128-d
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation="relu")(x)

    model = keras.Model(inputs, x, name="LocalExtractor")
    return model

# -------------------------
# 3. Dilated convolutional blocks for long-range features
# -------------------------
def dilated_block(x, dropout_rate=0.2):
    """
    A dilated block with 5 parallel Conv1D layers (dilation rates 2,4,8,16,32).
    Sums outputs, applies Dropout, adds residual connection.
    """
    conv_outputs = []
    for d_rate in [2, 4, 8, 16, 32]:
        y = layers.Conv1D(128, kernel_size=7, dilation_rate=d_rate, padding="same")(x)
        y = layers.LeakyReLU(alpha=0.01)(y)
        conv_outputs.append(y)

    y = layers.Add()(conv_outputs)
    y = layers.Dropout(dropout_rate)(y)
    y = layers.Add()([x, y])
    return y

def build_dilated_extractor(num_patches=960, embedding_dim=128, dropout_rate=0.2):
    """
    Processes the time-sequence of patch embeddings (shape (num_patches, 128)).
    Output shape remains (num_patches, 128) after two dilated blocks.
    """
    inputs = keras.Input(shape=(num_patches, embedding_dim))
    x = inputs
    x = dilated_block(x, dropout_rate=dropout_rate)
    x = dilated_block(x, dropout_rate=dropout_rate)
    model = keras.Model(inputs, x, name="DilatedExtractor")
    return model

# -------------------------
# 4. Full Model for 8-hour HR (2Hz => 57,600 samples)
# -------------------------
def build_full_model(
    input_length=57600,   # 8 hours * 3600 s/h * 2 Hz
    patch_length=256,
    patch_step=60,        # => 960 patches => (57600-256)/(960-1) ~ 60
    num_patches=960,
    num_classes=3,
    dropout_rate=0.2
):
    """
    - Input: (batch_size, 57600) => 8h of 2Hz heart rate
    - Frame into patches of length=256, step=60 => ~960 patches
    - Local Extractor => TimeDistributed => shape (batch, 960, 128)
    - Dilated Extractor => (batch, 960, 128)
    - Final Conv1D => shape (batch, 960, num_classes)
    """
    inputs = keras.Input(shape=(input_length,), name="IHR_input")

    # Use a custom layer to do tf.signal.frame on the symbolic input
    framer = FrameLayer(
        frame_length=patch_length,
        frame_step=patch_step,
        num_patches=num_patches
    )(inputs)
    # framer => shape (batch, num_patches, patch_length)

    # Expand dims for channel
    patches = layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(framer)
    # => shape (batch, num_patches, patch_length, 1)

    # Local feature extraction
    local_extractor = build_local_extractor(input_shape=(patch_length, 1))
    embeddings = layers.TimeDistributed(local_extractor)(patches)
    # => shape (batch, num_patches, 128)

    # Dilated conv blocks
    dilated_extractor = build_dilated_extractor(
        num_patches=num_patches,
        embedding_dim=128,
        dropout_rate=dropout_rate
    )
    x = dilated_extractor(embeddings)
    # => shape (batch, num_patches, 128)

    # Final classification layer => shape (batch, num_patches, num_classes)
    outputs = layers.Conv1D(num_classes, kernel_size=1, padding="same", activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="SleepStageDilatedCNN_8H")
    return model

# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    num_classes = 3  # e.g. 5 classes
    model = build_full_model(
        input_length=57600,
        patch_length=256,
        patch_step=59,
        num_patches=960,
        num_classes=num_classes,
        dropout_rate=0.2
    )
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()

