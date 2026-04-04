"""
Lightweight CNN for noise estimation (OPTIONAL).

BEHAVIOR:
- If TensorFlow is NOT installed: module silently fails to import, pipeline uses heuristic estimators (safe fallback).
- If TensorFlow IS installed:
  - If model path provided: loads pretrained model and uses it to refine noise estimate.
  - If model path NOT provided: builds a fresh untrained model (random predictions until trained).
- In all cases, the pipeline continues safely: CNN refinement is a 'nice-to-have' improvement, not required.

TRAINING:
- To use this effectively, you need a trained model (see train_noise_cnn.py if provided).
- OR: Install TensorFlow + skip CNN (rely on heuristic estimators which work well).
- OR: Keep `load_noise_model(path=None)` and let it build a fresh model (no effect until trained).

The model is tiny: 3 conv layers, batch norm, global avg pool, one dense output.
"""

import os
import numpy as np

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False


def build_small_noise_model(input_shape=(64, 64, 1)):
    if not TF_AVAILABLE:
        raise RuntimeError('TensorFlow not available')
    inp = layers.Input(shape=input_shape)
    x = layers.Conv2D(16, 3, activation='relu', padding='same')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D(2)(x)
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D(2)(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(32, activation='relu')(x)
    out = layers.Dense(1, activation='linear')(x)
    m = models.Model(inp, out)
    m.compile(optimizer='adam', loss='mse')
    return m


def load_noise_model(path=None, input_shape=(64,64,1)):
    """Load a pretrained model if `path` provided, else look for noise_model.h5, else build fresh.
    Returns None if TF unavailable.
    """
    if not TF_AVAILABLE:
        return None
    
    # Try explicit path first
    if path is not None:
        try:
            return tf.keras.models.load_model(path)
        except Exception:
            pass
    
    # Try default trained model path
    default_model_path = 'noise_model.h5'
    if os.path.exists(default_model_path):
        try:
            return tf.keras.models.load_model(default_model_path)
        except Exception:
            pass
    
    # Fallback: build fresh (untrained)
    return build_small_noise_model(input_shape)


def estimate_noise_cnn(img, model, input_size=64):
    """Estimate noise level from `img` (BGR uint8) using `model`.
    Returns float noise estimate or None if model not provided.
    """
    if model is None or not TF_AVAILABLE:
        return None
    # Convert to grayscale and center-crop/resize
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h, w = gray.shape[:2]
    # Resize to input_size
    import cv2
    small = cv2.resize(gray, (input_size, input_size), interpolation=cv2.INTER_AREA)
    arr = small.astype(np.float32) / 255.0
    arr = arr.reshape((1, input_size, input_size, 1))
    pred = model.predict(arr, verbose=0)
    return float(pred.flatten()[0])
