"""
Quick training script for lightweight noise CNN.
Trains on your images by adding synthetic noise at different levels.
Takes ~5 minutes on a modern CPU/GPU.

Usage:
    python train_noise_cnn.py --dataset dataset/old_images --epochs 10 --output noise_model.h5
"""

import os
import argparse
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from noise_cnn import build_small_noise_model


def load_images_from_folder(folder, max_images=50, img_size=64):
    """Load images from folder and resize to img_size."""
    images = []
    for fname in os.listdir(folder)[:max_images]:
        if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            path = os.path.join(folder, fname)
            try:
                img = cv2.imread(path)
                if img is not None:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    small = cv2.resize(gray, (img_size, img_size), interpolation=cv2.INTER_AREA)
                    images.append(small.astype(np.float32) / 255.0)
            except Exception as e:
                print(f'Warning: could not load {fname}: {e}')
    return np.array(images)


def generate_noisy_pairs(clean_images, noise_levels=None, samples_per_level=3):
    """Generate (noisy_image, noise_level) pairs from clean images."""
    if noise_levels is None:
        noise_levels = np.linspace(5, 50, 10)  # 10 noise levels from 5 to 50
    
    X_train = []
    y_train = []
    
    for clean_img in clean_images:
        for noise_level in noise_levels:
            for _ in range(samples_per_level):
                # Add Gaussian noise
                noise = np.random.normal(0, noise_level / 255.0, clean_img.shape)
                noisy = np.clip(clean_img + noise, 0, 1)
                X_train.append(noisy)
                y_train.append(noise_level)
    
    return np.array(X_train), np.array(y_train)


def train_noise_model(dataset_folder, epochs=10, output_path='noise_model.h5', img_size=64):
    """Train the noise CNN on synthetic noisy data derived from real images."""
    
    print('Loading images from:', dataset_folder)
    clean_images = load_images_from_folder(dataset_folder, max_images=50, img_size=img_size)
    if len(clean_images) == 0:
        print('ERROR: No images found in', dataset_folder)
        return
    
    print(f'Loaded {len(clean_images)} images.')
    
    print('Generating synthetic noisy training data...')
    X_train, y_train = generate_noisy_pairs(clean_images, noise_levels=np.linspace(5, 50, 10), samples_per_level=3)
    X_train = X_train.reshape((X_train.shape[0], img_size, img_size, 1))
    
    print(f'Training set: {X_train.shape} -> {y_train.shape}')
    
    print('Building model...')
    model = build_small_noise_model(input_shape=(img_size, img_size, 1))
    
    print('Training...')
    history = model.fit(
        X_train, y_train,
        batch_size=16,
        epochs=epochs,
        validation_split=0.2,
        verbose=1
    )
    
    print(f'Saving model to {output_path}...')
    model.save(output_path)
    print('✓ Model trained and saved. Pipeline will auto-load it on next run.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train noise CNN on synthetic data')
    parser.add_argument('--dataset', '-d', default='dataset/old_images',
                        help='Folder containing training images')
    parser.add_argument('--epochs', '-e', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--output', '-o', default='noise_model.h5',
                        help='Output model file path')
    parser.add_argument('--img-size', type=int, default=64,
                        help='Input image size for training')
    args = parser.parse_args()
    
    train_noise_model(args.dataset, epochs=args.epochs, output_path=args.output, img_size=args.img_size)
