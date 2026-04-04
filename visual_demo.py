"""
Visual demonstration — zoomed + step-by-step output.
Run: python visual_demo.py --input dataset/old_images/old_rose.png
"""
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import os
from restoration import (restore_image, white_balance_adaptive,
                          enhance_contrast_multiscale, detect_spots_mask,
                          suppress_fold_lines, adaptive_unsharp_mask,
                          detect_blur_level)

def save_step_by_step(img, out_path):
    """Save 6-panel step-by-step pipeline visualization."""
    blur_level, _ = detect_blur_level(img)

    # Step by step
    step1 = img.copy()
    step2, _, _ = white_balance_adaptive(step1)
    _, have_spots = detect_spots_mask(step2)
    step3 = cv2.inpaint(step2, detect_spots_mask(step2)[0], 2,
                        cv2.INPAINT_TELEA) if have_spots else step2
    step4, _, _ = suppress_fold_lines(step3)
    step5 = enhance_contrast_multiscale(step4, clip_limit=1.2)
    step6 = adaptive_unsharp_mask(step5, blur_level, base_amount=0.35)

    steps  = [step1, step2, step3, step4, step5, step6]
    titles = ['1. Original', '2. White Balance',
              '3. Spot Removal', '4. Fold Suppression',
              '5. CLAHE Enhancement', '6. Final Restored']

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), dpi=100)
    axes = axes.flatten()
    for i, (s, t) in enumerate(zip(steps, titles)):
        axes[i].imshow(cv2.cvtColor(s, cv2.COLOR_BGR2RGB))
        axes[i].set_title(t, fontsize=12, fontweight='bold')
        axes[i].axis('off')

    plt.suptitle('Pipeline Step-by-Step Visualization', fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f'Step-by-step saved: {out_path}')


def save_zoomed_comparison(img, restored, out_path, region_frac=0.35):
    """Save zoomed-in region comparison side by side."""
    h, w   = img.shape[:2]
    y1     = int(h * 0.30)
    y2     = int(h * (0.30 + region_frac))
    x1     = int(w * 0.30)
    x2     = int(w * (0.30 + region_frac))

    crop_orig = img[y1:y2, x1:x2]
    crop_rest = restored[y1:y2, x1:x2]

    # Draw red rectangle on full images showing zoom area
    full_orig = img.copy()
    full_rest = restored.copy()
    cv2.rectangle(full_orig, (x1, y1), (x2, y2), (0, 0, 255), 3)
    cv2.rectangle(full_rest, (x1, y1), (x2, y2), (0, 0, 255), 3)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=100)
    axes[0][0].imshow(cv2.cvtColor(full_orig, cv2.COLOR_BGR2RGB))
    axes[0][0].set_title('Original (full)', fontsize=12)
    axes[0][0].axis('off')

    axes[0][1].imshow(cv2.cvtColor(full_rest, cv2.COLOR_BGR2RGB))
    axes[0][1].set_title('Restored (full)', fontsize=12)
    axes[0][1].axis('off')

    axes[1][0].imshow(cv2.cvtColor(crop_orig, cv2.COLOR_BGR2RGB))
    axes[1][0].set_title('Original (zoomed)', fontsize=12, color='red')
    axes[1][0].axis('off')

    axes[1][1].imshow(cv2.cvtColor(crop_rest, cv2.COLOR_BGR2RGB))
    axes[1][1].set_title('Restored (zoomed)', fontsize=12, color='red')
    axes[1][1].axis('off')

    plt.suptitle('Zoomed Region Comparison', fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f'Zoomed comparison saved: {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',  '-i', required=True)
    parser.add_argument('--output', '-o', default='results/visual_demo')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    img      = cv2.imread(args.input)
    fname    = os.path.splitext(os.path.basename(args.input))[0]
    restored = restore_image(img, sat_scale=1.5, nlm_h=7, median_k=3,
                             clahe_clip=1.2, sat_scale_override=1.5,
                             unsharp_amount=0.35)

    save_step_by_step(img,
        os.path.join(args.output, f'{fname}_steps.png'))
    save_zoomed_comparison(img, restored,
        os.path.join(args.output, f'{fname}_zoomed.png'))