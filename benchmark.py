"""
benchmark.py

Simple benchmarking utility to measure per-step runtime of the restoration pipeline.

Usage:
    python benchmark.py --input dataset/old_images/old.png --output results/benchmark.json --repeats 3

It calls analysis + restore steps and measures wall-clock time for each named stage.
"""
import argparse
import json
import os
import time
import cv2
from restoration import (
    analyze_and_restore, restore_image, detect_blur_level, estimate_noise,
    nl_means_denoise, white_balance_adaptive, detect_spots_mask, inpaint_spots,
    suppress_fold_lines, enhance_contrast_multiscale, increase_saturation,
    adaptive_unsharp_mask
)


def time_call(fn, *args, repeats=1, **kwargs):
    t0 = time.perf_counter()
    for _ in range(repeats):
        _ = fn(*args, **kwargs)
    t1 = time.perf_counter()
    return (t1 - t0) / float(repeats)


def run_benchmark(img_path, out_path, repeats=1):
    img = cv2.imread(img_path)
    assert img is not None, f'Could not read {img_path}'

    results = {}

    # Analysis
    results['analysis'] = time_call(detect_blur_level, img, repeats=repeats)
    results['estimate_noise'] = time_call(estimate_noise, img, repeats=repeats)

    # Denoise (NLM) - run on a copy
    results['nlm_denoise'] = time_call(nl_means_denoise, img.copy(),  h=6, hColor=6, repeats=repeats)

    # Adaptive white balance
    results['white_balance_adaptive'] = time_call(white_balance_adaptive, img.copy(), repeats=repeats)

    # Spot detection + inpaint
    mask = detect_spots_mask(img.copy())[0]
    results['detect_spots_mask'] = time_call(detect_spots_mask, img.copy(), repeats=repeats)
    results['inpaint_spots'] = time_call(inpaint_spots, img.copy(), mask, repeats=repeats)

    # Fold suppression
    results['suppress_fold_lines'] = time_call(suppress_fold_lines, img.copy(), repeats=repeats)

    # Contrast (multi-scale CLAHE)
    results['enhance_contrast_multiscale'] = time_call(enhance_contrast_multiscale, img.copy(), repeats=repeats)

    # Saturation + unsharp
    results['increase_saturation'] = time_call(increase_saturation, img.copy(), scale=1.5, repeats=repeats)
    results['adaptive_unsharp_mask'] = time_call(adaptive_unsharp_mask, img.copy(), 200, repeats=repeats)

    # Full restore
    results['restore_image'] = time_call(restore_image, img.copy(), repeats=repeats)

    # Optional: deblur (light test)
    try:
        results['deblur_image'] = time_call(deblur_image, img.copy(), 21, 3.0, iterations=8, repeats=repeats)
    except Exception:
        results['deblur_image'] = None

    # Save results
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    print('Benchmark results saved to', out_path)
    for k, v in results.items():
        print(f'{k:30s}: {v}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark restoration pipeline steps')
    parser.add_argument('--input', '-i', default=os.path.join('dataset', 'old_images', 'old.png'))
    parser.add_argument('--output', '-o', default=os.path.join('results', 'benchmark.json'))
    parser.add_argument('--repeats', '-r', type=int, default=1)
    args = parser.parse_args()

    run_benchmark(args.input, args.output, repeats=args.repeats)
