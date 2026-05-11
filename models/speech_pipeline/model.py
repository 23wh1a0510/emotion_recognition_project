"""Model definition: Conv1D + Bidirectional LSTM for speech emotion recognition.
"""
from typing import Tuple
import tensorflow as tf


def build_model(input_shape: Tuple[int, int], num_classes: int, dropout: float = 0.3) -> tf.keras.Model:
    """Builds and returns a Keras model.

    input_shape: (time_steps, n_features)
    """
    inputs = tf.keras.Input(shape=input_shape, name='input')

    # Conv block 1
    x = tf.keras.layers.Conv1D(filters=64, kernel_size=5, activation='relu', padding='same')(inputs)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = tf.keras.layers.Dropout(dropout)(x)

    # Conv block 2
    x = tf.keras.layers.Conv1D(filters=128, kernel_size=5, activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = tf.keras.layers.Dropout(dropout)(x)

    # BiLSTM
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=False))(x)
    x = tf.keras.layers.Dropout(dropout)(x)

    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(dropout)(x)

    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', name='output')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='conv1d_bilstm')
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
