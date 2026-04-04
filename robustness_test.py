"""
Robustness test — synthesize 4 degradation types and measure pipeline performance.
Run: python robustness_test.py --input dataset/old_images/old_rose.png
"""
import cv2
import numpy as np
import argparse
import logging
from restoration import restore_image, brisque_score, niqe_score, psnr, ssim

def add_gaussian_noise(img, sigma=25):
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

def add_fading(img, alpha=0.5, beta=30):
    """Simulate color fading — reduce contrast + add yellow tint."""
    faded = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    faded[:, :, 0] = np.clip(faded[:, :, 0].astype(int) - 20, 0, 255)  # reduce blue
    return faded

def add_fold_line(img):
    """Add a synthetic horizontal fold line."""
    result = img.copy()
    h, w   = result.shape[:2]
    y      = h // 2
    cv2.line(result, (0, y), (w, y), (200, 200, 200), 4)
    return result

def add_low_contrast(img):
    """Compress pixel range to simulate low contrast."""
    return cv2.convertScaleAbs(img, alpha=0.4, beta=80)

def run_robustness_test(img_path, output_dir='results/robustness'):
    import os
    os.makedirs(output_dir, exist_ok=True)

    img = cv2.imread(img_path)
    if img is None:
        print('Cannot read:', img_path)
        return

    degradations = {
        'noisy':        add_gaussian_noise(img, sigma=25),
        'faded':        add_fading(img),
        'folded':       add_fold_line(img),
        'low_contrast': add_low_contrast(img),
    }

    orig_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print('\n' + '='*85)
    print(f"{'Degradation':<20} {'BRISQUE Before':>15} {'BRISQUE After':>14} "
          f"{'NIQE Before':>12} {'NIQE After':>11} {'PSNR':>8}")
    print('='*85)

    for name, degraded in degradations.items():
        restored = restore_image(degraded, sat_scale=1.5, nlm_h=7,
                                 median_k=3, clahe_clip=1.2,
                                 sat_scale_override=1.5, unsharp_amount=0.35)

        b_before = brisque_score(degraded)
        b_after  = brisque_score(restored)
        n_before = niqe_score(degraded)
        n_after  = niqe_score(restored)

        res_gray = cv2.cvtColor(restored, cv2.COLOR_BGR2GRAY)
        p_val    = psnr(orig_gray, res_gray)

        print(f"{name:<20} {b_before:>15.4f} {b_after:>14.4f} "
              f"{n_before:>12.4f} {n_after:>11.4f} {p_val:>8.2f}")

        # Save before/after
        cv2.imwrite(f'{output_dir}/{name}_degraded.png',  degraded)
        cv2.imwrite(f'{output_dir}/{name}_restored.png',  restored)

    print('='*85)
    print('Lower BRISQUE/NIQE after = pipeline handles this degradation well\n')
    print(f'Images saved to: {output_dir}/')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', required=True)
    parser.add_argument('--output', '-o', default='results/robustness')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    run_robustness_test(args.input, args.output)