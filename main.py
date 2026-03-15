"""
Color Restoration Application

Required libraries:
- OpenCV: pip install opencv-python
- NumPy: pip install numpy
- Matplotlib: pip install matplotlib

How to run:
1. Place your input images (jpg/png/bmp/tiff) into:
   color_restoration_project/dataset/old_images/
2. Run:
   python main.py
3. Restored images will be saved to:
   color_restoration_project/results/restored_images/

This script reads all images from the input folder, applies a
restoration pipeline (see restoration.py), displays each original
and restored image side-by-side, and saves the restored output.
"""

import os
import cv2
import matplotlib.pyplot as plt
from restoration import restore_image, analyze_and_restore


def process_all(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')

    files = [f for f in os.listdir(input_dir) if f.lower().endswith(exts)]
    if not files:
        print('No images found in', input_dir)
        return

    for fname in sorted(files):
        in_path = os.path.join(input_dir, fname)
        print('Processing:', in_path)
        img = cv2.imread(in_path)
        if img is None:
            print('  Skipping (cannot read):', fname)
            continue

        # Adaptive pipeline: analyze and choose operations
        restored, info = analyze_and_restore(img)

        out_name = f'restored_{fname}'
        out_path = os.path.join(output_dir, out_name)
        cv2.imwrite(out_path, restored)
        print('  Saved restored image to:', out_path)

        # Print detected condition and metrics
        cond = 'Noisy' if info.get('is_noisy') else 'Clean'
        if info.get('is_low_contrast'):
            cond += ' + Low-Contrast'
        if info.get('is_grayscale'):
            cond += ' + Grayscale'
        print('  Detected condition:', cond)
        print(f"  Noise level: {info.get('noise_level'):.2f}, Contrast score: {info.get('contrast_score'):.2f}")
        print(f"  MSE: {info.get('mse'):.2f}, PSNR: {info.get('psnr'):.2f}, SSIM: {info.get('ssim'):.4f}")

        # Display original vs restored side-by-side
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original Image')
        axes[1].imshow(cv2.cvtColor(restored, cv2.COLOR_BGR2RGB))
        axes[1].set_title('Restored Image')
        for ax in axes:
            ax.axis('off')
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(base_dir, 'dataset', 'old_images')
    output_dir = os.path.join(base_dir, 'results', 'restored_images')

    # Ensure folders exist
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    print('Input folder:', input_dir)
    print('Output folder:', output_dir)

    process_all(input_dir, output_dir)
