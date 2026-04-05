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

New features:
- Adaptive white balance using Hasler-Suesstrunk colorfulness metric
- Fold line suppression using Hough Transform
- Multi-Scale CLAHE (3 tile sizes blended)
- Continuous adaptation — replaces hard if/elif blur branches     [#9]
- Data-driven parameter optimization using BRISQUE                [#10]
- Image difficulty-aware processing (low/medium/severe)           [#11]
- Improved noise estimation (patch + frequency domain)            [#12]
- Ablation study: run with --ablation flag
- Debug mode: run with --debug flag to save fold/spot overlays
"""

import os
import argparse
import logging
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend — fixes MemoryError on large images
import matplotlib.pyplot as plt

# Try to import CNN noise model (optional — will fall back to heuristic if unavailable)
try:
    from noise_cnn import load_noise_model
    CNN_AVAILABLE = True
except Exception:
    CNN_AVAILABLE = False

from restoration import (restore_image, analyze_and_restore,
                          run_ablation_study, print_ablation_table,
                          mse, psnr, ssim, colorfulness_metric, adaptive_wb_weight,
                          detect_blur_level, detect_fold_lines, detect_spots_mask,
                          save_debug_overlays,
                          classify_noise_type, entropy_metric, adaptive_wb_weight_entropy,
                          optimize_parameters,           # #10
                          estimate_noise_advanced,       # #12
                          contrast_score,
                          estimate_noise)                # #21


# Maximum image side length for in-memory processing
MAX_SIDE = 1200


# ── Improvement #9: Continuous Adaptation ─────────────────────────────────────
def compute_params_continuous(blur_level):
    """Smoothly interpolate parameters based on blur level — no hard jumps.

    blur_level close to 0   → very blurry  → aggressive settings
    blur_level close to 500 → very sharp   → gentle settings
    """
    sharpness = float(np.clip(blur_level / 500.0, 0.0, 1.0))

    params = dict(
        nlm_h              = int(round(8 - 2 * sharpness)),   # 8 (blurry) → 6 (sharp)
        median_k           = 3,
        clahe_clip         = round(1.5 - 0.4 * sharpness, 2), # 1.5 → 1.1
        sat_scale_override = 1.5,
        unsharp_amount     = round(0.5 - 0.2 * sharpness, 2), # 0.5 → 0.3
        spot_thresh        = 40,
        inpaint_radius     = 2,
        use_fold_suppression = True,
        use_multiscale_clahe = True,
        use_deblur           = sharpness < 0.5,
        use_adaptive_sharpen = True,
    )

    logging.info('--- Continuous Adaptation (#9) ---')
    logging.info('  Sharpness score : %.3f  (0=blurry, 1=sharp)', sharpness)
    logging.info('  nlm_h           : %d',   params['nlm_h'])
    logging.info('  clahe_clip      : %.2f', params['clahe_clip'])
    logging.info('  unsharp_amount  : %.2f', params['unsharp_amount'])
    logging.info('  use_deblur      : %s',   params['use_deblur'])

    return params


# ── Improvement #11: Image Difficulty-Aware Processing ────────────────────────
def difficulty_score(img):
    """Compute composite difficulty score from 4 signals.

    Returns (score 0-1, level, noise_norm, contrast_norm, blur_norm, color_norm).
      0.00 - 0.33 → low    (mild degradation)
      0.33 - 0.66 → medium
      0.66 - 1.00 → severe
    """
    noise_lvl   = estimate_noise(img)
    cont        = contrast_score(img)
    blur_lvl, _ = detect_blur_level(img)
    cf          = colorfulness_metric(img)

    # Normalize each to [0,1] — higher = worse / more degraded
    noise_norm    = float(np.clip(noise_lvl  / 30.0,       0.0, 1.0))
    contrast_norm = float(np.clip(1.0 - cont / 60.0,       0.0, 1.0))
    blur_norm     = float(np.clip(1.0 - blur_lvl / 500.0,  0.0, 1.0))
    color_norm    = float(np.clip(1.0 - cf / 50.0,         0.0, 1.0))

    score = (0.30 * noise_norm +
             0.25 * contrast_norm +
             0.25 * blur_norm +
             0.20 * color_norm)
    score = float(np.clip(score, 0.0, 1.0))

    level = 'low' if score < 0.33 else ('medium' if score < 0.66 else 'severe')

    logging.info('--- Difficulty-Aware Processing (#11) ---')
    logging.info('  Noise norm    : %.2f', noise_norm)
    logging.info('  Contrast norm : %.2f', contrast_norm)
    logging.info('  Blur norm     : %.2f', blur_norm)
    logging.info('  Color norm    : %.2f', color_norm)
    logging.info('  Difficulty    : %.2f  →  Level: %s', score, level)

    return score, level, noise_norm, contrast_norm, blur_norm, color_norm


def intensity_from_difficulty(level):
    """Map difficulty level to pipeline intensity multipliers."""
    presets = {
        'low':    dict(nlm_h=5,  clahe_clip=1.0, sat_scale_override=1.3, unsharp_amount=0.20),
        'medium': dict(nlm_h=7,  clahe_clip=1.2, sat_scale_override=1.5, unsharp_amount=0.35),
        'severe': dict(nlm_h=10, clahe_clip=1.5, sat_scale_override=1.7, unsharp_amount=0.50),
    }
    return presets[level]


# ── CNN helper ────────────────────────────────────────────────────────────────
def restore_with_cnn(img, model_path=None):
    """Restore image using CNN-based noise estimation + heuristic restoration.
    Falls back to heuristic if CNN unavailable.
    """
    if not CNN_AVAILABLE:
        logging.warning('CNN not available — falling back to heuristic restoration')
        return restore_image(img, nlm_h=6, median_k=3, clahe_clip=1.1,
                             sat_scale_override=1.5, unsharp_amount=0.3)
    try:
        model = load_noise_model(path=model_path)
        if model is None:
            logging.warning('Could not load CNN model — using heuristic')
            return restore_image(img, nlm_h=6, median_k=3, clahe_clip=1.1,
                                 sat_scale_override=1.5, unsharp_amount=0.3)

        gray       = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        patch_size = 64
        stride     = 32
        patches    = []

        for y in range(0, gray.shape[0] - patch_size, stride):
            for x in range(0, gray.shape[1] - patch_size, stride):
                patch = gray[y:y+patch_size, x:x+patch_size].astype(np.float32) / 255.0
                patches.append(patch[np.newaxis, :, :, np.newaxis])

        avg_noise = float(np.mean(model.predict(
            np.concatenate(patches, axis=0), verbose=0))) if patches else 15.0

        logging.info('CNN-estimated noise level: %.2f', avg_noise)

        nlm_h          = int(np.clip(4 + (avg_noise / 50.0) * 6,   4,   10))
        clahe_clip     = float(np.clip(1.0 + (avg_noise / 50.0) * 0.5, 1.0, 1.5))
        unsharp_amount = float(np.clip(0.2 + (avg_noise / 50.0) * 0.4, 0.2, 0.6))

        logging.info('CNN-adapted params: nlm_h=%d, clahe_clip=%.2f, unsharp=%.2f',
                     nlm_h, clahe_clip, unsharp_amount)

        return restore_image(img, nlm_h=nlm_h, median_k=3, clahe_clip=clahe_clip,
                             sat_scale_override=1.5, unsharp_amount=unsharp_amount)
    except Exception as e:
        logging.warning('CNN restoration failed: %s — falling back to heuristic', str(e))
        return restore_image(img, nlm_h=6, median_k=3, clahe_clip=1.1,
                             sat_scale_override=1.5, unsharp_amount=0.3)


# ── Improvement #21: Failure Case Analysis ────────────────────────────────────
def detect_failure_case(info, mild_variant, img):
    """Detect if the restoration likely failed or produced poor results."""
    reasons = []

    if info.get('ssim', 1.0) < 0.35:
        reasons.append(f"Very low SSIM ({info['ssim']:.3f}) — structural damage")

    if info.get('psnr', 99.0) < 15.0:
        reasons.append(f"Low PSNR ({info['psnr']:.2f} dB) — large pixel deviation")

    if info.get('blur_level', 999) < 30:
        reasons.append(f"Extreme blur ({info['blur_level']:.1f}) — beyond classical recovery")

    try:
        post_noise = estimate_noise(mild_variant)
        if post_noise > 15.0:
            reasons.append(f"High residual noise ({post_noise:.1f}) after denoising")
    except Exception:
        pass

    return reasons


# ── Shared helper: compute #10, #11, #12 for any mode ─────────────────────────
def _compute_all_diagnostics(img_proc, mild_params, blur_level):
    """Always compute #10, #11, #12 values regardless of pipeline mode.
    This ensures the comparison image box always shows real values.
    Returns (opt_params, opt_score, diff_score, diff_level, diff_intensity,
             patch_noise, freq_noise, combined_noise, noise_decision, sharpness)
    """
    sharpness = float(np.clip(blur_level / 500.0, 0.0, 1.0))

    # ── #10: Data-Driven Optimization ─────────────────────────────────────
    try:
        opt_params, opt_score = optimize_parameters(img_proc)
        logging.info('--- Data-Driven Optimization (#10) ---')
        logging.info('  Best wb_weight  : %.2f', opt_params['wb_weight'])
        logging.info('  Best sat_scale  : %.2f', opt_params['sat_scale'])
        logging.info('  Best clahe_clip : %.2f', opt_params['clahe_clip'])
        logging.info('  Best BRISQUE    : %.4f', opt_score)
    except Exception:
        opt_params = {'wb_weight': 0.5, 'sat_scale': 1.5, 'clahe_clip': 1.2}
        opt_score  = 0.0
        logging.warning('Optimization (#10) failed — using defaults')

    # ── #11: Difficulty-Aware ──────────────────────────────────────────────
    try:
        diff_score, diff_level, n_norm, c_norm, b_norm, col_norm = \
            difficulty_score(img_proc)
        diff_intensity = intensity_from_difficulty(diff_level)
    except Exception:
        diff_score     = 0.0
        diff_level     = 'unknown'
        diff_intensity = {
            'nlm_h':          mild_params.get('nlm_h', 6),
            'clahe_clip':     mild_params.get('clahe_clip', 1.2),
            'sat_scale_override': mild_params.get('sat_scale_override', 1.5),
            'unsharp_amount': mild_params.get('unsharp_amount', 0.3),
        }
        logging.warning('Difficulty score (#11) failed — using defaults')

    # ── #12: Advanced Noise Estimation ────────────────────────────────────
    try:
        patch_noise, freq_noise, combined_noise = estimate_noise_advanced(img_proc)
        noise_decision = 'NLM Denoise' if combined_noise > 10.0 else 'Median Blur'
        logging.info('--- Advanced Noise Estimation (#12) ---')
        logging.info('  Patch-based noise : %.2f', patch_noise)
        logging.info('  Frequency noise   : %.2f', freq_noise)
        logging.info('  Combined estimate : %.2f', combined_noise)
        logging.info('  Decision          : %s', noise_decision)
    except Exception:
        patch_noise = freq_noise = combined_noise = 0.0
        noise_decision = '-'
        logging.warning('Noise estimation (#12) failed — using defaults')

    return (opt_params, opt_score,
            diff_score, diff_level, diff_intensity,
            patch_noise, freq_noise, combined_noise, noise_decision,
            sharpness)


# ── Main processing loop ───────────────────────────────────────────────────────
def process_all(input_dir, output_dir, display=True, run_ablation=False,
                debug=False, single_file=None, mode='heuristic',
                run_benchmark=False):
    os.makedirs(output_dir, exist_ok=True)

    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')

    if single_file:
        if not os.path.exists(single_file):
            print('File not found:', single_file)
            return
        input_dir = os.path.dirname(single_file) or input_dir
        files = [os.path.basename(single_file)]
    else:
        files = [f for f in os.listdir(input_dir) if f.lower().endswith(exts)]
        if not files:
            print('No images found in', input_dir)
            return

    for fname in sorted(files):
        in_path = os.path.join(input_dir, fname)
        logging.info('Processing: %s', in_path)
        try:
            img = cv2.imread(in_path)
            if img is None:
                logging.warning('Skipping (cannot read): %s', fname)
                continue

            # Downscale very large images to reduce memory footprint
            h, w  = img.shape[:2]
            scale = 1.0
            if max(h, w) > MAX_SIDE:
                scale    = float(MAX_SIDE) / float(max(h, w))
                new_w    = int(w * scale)
                new_h    = int(h * scale)
                img_proc = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                logging.info('Downscaled %s -> %dx%d for processing', fname, new_w, new_h)
            else:
                img_proc = img

            # Analyze image
            try:
                blur_level, is_blurred = detect_blur_level(img_proc)
                _, info = analyze_and_restore(img_proc)
                info['scale_factor'] = scale
            except MemoryError:
                logging.warning('MemoryError during analysis — retrying at 50%% scale')
                try:
                    img_proc = cv2.resize(img_proc,
                                          (max(64, int(img_proc.shape[1] // 2)),
                                           max(64, int(img_proc.shape[0] // 2))),
                                          interpolation=cv2.INTER_AREA)
                    blur_level, is_blurred = detect_blur_level(img_proc)
                    _, info = analyze_and_restore(img_proc)
                    info['scale_factor'] = scale * 0.5
                except Exception:
                    logging.exception('Processing failed for %s', in_path)
                    continue

            # ── Debug overlay ──────────────────────────────────────────────
            if debug:
                try:
                    folds = detect_fold_lines(img_proc)
                    spots_mask, _ = detect_spots_mask(img_proc)
                    debug_prefix = os.path.join(
                        output_dir, f'debug_{os.path.splitext(fname)[0]}')
                    save_debug_overlays(img_proc, folds, spots_mask, debug_prefix)
                    logging.info('Saved debug overlays for: %s', debug_prefix)
                except Exception:
                    logging.exception('Failed to save debug overlay for %s', fname)

            # ── Choose pipeline mode & run restoration ─────────────────────
            if mode == 'cnn':
                logging.info('Using CNN-based restoration')
                mild_params = {'nlm_h': 6, 'clahe_clip': 1.1,
                               'unsharp_amount': 0.3, 'sat_scale_override': 1.5}
                mild_variant = restore_with_cnn(img_proc, model_path=os.path.join(
                    os.path.dirname(__file__), 'noise_model.h5'))

            elif mode == 'hybrid':
                logging.info('Using hybrid restoration (heuristic + CNN)')
                mild_params      = compute_params_continuous(blur_level)
                heuristic_result = restore_image(img_proc, **mild_params)
                cnn_result       = restore_with_cnn(img_proc, model_path=os.path.join(
                    os.path.dirname(__file__), 'noise_model.h5'))
                mild_variant     = cv2.addWeighted(heuristic_result, 0.5,
                                                   cnn_result, 0.5, 0)

            elif mode == 'difficulty':
                logging.info('Using difficulty-aware restoration')
                # Compute difficulty first, use its intensity as params
                d_score, d_level, *_ = difficulty_score(img_proc)
                mild_params  = intensity_from_difficulty(d_level)
                mild_params.setdefault('median_k',           3)
                mild_params.setdefault('spot_thresh',        40)
                mild_params.setdefault('inpaint_radius',     2)
                mild_params.setdefault('use_fold_suppression', True)
                mild_params.setdefault('use_multiscale_clahe', True)
                mild_params.setdefault('use_deblur',           True)
                mild_params.setdefault('use_adaptive_sharpen', True)
                mild_variant = restore_image(img_proc, **mild_params)

            elif mode == 'legacy':
                logging.info('Using legacy (continuous + BRISQUE) restoration')
                mild_params = compute_params_continuous(blur_level)
                try:
                    opt_p, _ = optimize_parameters(img_proc)
                    mild_params['clahe_clip'] = opt_p['clahe_clip']
                except Exception:
                    pass
                mild_variant = restore_image(img_proc, **mild_params)

            else:  # 'heuristic' — default
                logging.info('Using heuristic restoration')
                mild_params  = compute_params_continuous(blur_level)
                mild_variant = restore_image(img_proc, **mild_params)

            # ── Always compute #10, #11, #12 diagnostics for display ───────
            # This runs AFTER restoration so it never blocks the pipeline.
            # All modes get real values in the comparison box.
            (opt_params, opt_score,
             diff_score, diff_level, diff_intensity,
             patch_noise, freq_noise, combined_noise, noise_decision,
             sharpness) = _compute_all_diagnostics(img_proc, mild_params, blur_level)

            # Save restored image
            out_name = f'restored_{fname}'
            out_path = os.path.join(output_dir, out_name)
            cv2.imwrite(out_path, mild_variant)
            logging.info('Saved restored image to: %s', out_path)

            # Compute metrics
            try:
                target_h, target_w = mild_variant.shape[:2]
                orig_resized = cv2.resize(img, (target_w, target_h),
                                          interpolation=cv2.INTER_AREA)
                orig_gray = cv2.cvtColor(orig_resized, cv2.COLOR_BGR2GRAY)
                mild_gray = cv2.cvtColor(mild_variant, cv2.COLOR_BGR2GRAY)
                info['mse']  = mse(orig_gray,  mild_gray)
                info['psnr'] = psnr(orig_gray, mild_gray)
                info['ssim'] = ssim(orig_gray, mild_gray)
            except Exception:
                logging.exception('Failed to compute metrics')

            # Build condition string
            cond = 'Noisy' if info.get('is_noisy') else 'Clean'
            if info.get('is_low_contrast'):
                cond += ' + Low-Contrast'
            if info.get('is_grayscale'):
                cond += ' + Grayscale'
            if info.get('is_blurred'):
                cond += ' + Blurred'

            cf       = info.get('colorfulness', 0.0)
            wb_w     = info.get('wb_weight_used', 0.0)
            blur_lvl = info.get('blur_level', 0.0)
            ntype    = info.get('noise_type', 'gaussian')
            ent      = info.get('entropy', 0.0)

            logging.info('Detected condition  : %s', cond)
            logging.info('Blur level          : %.2f  (Blurry threshold: 200)', blur_lvl)
            logging.info('Colorfulness        : %.2f  →  WB weight used: %.2f', cf, wb_w)
            logging.info('Noise type          : %s    Entropy: %.2f', ntype, ent)
            logging.info('Noise level         : %.2f,  Contrast score: %.2f',
                         info.get('noise_level', 0), info.get('contrast_score', 0))
            logging.info('MSE: %.2f,  PSNR: %.2f dB,  SSIM: %.4f',
                         info.get('mse', 0), info.get('psnr', 0), info.get('ssim', 0))

            # ── Improvement #21: Failure Case Analysis ─────────────────────
            failure_reasons = detect_failure_case(info, mild_variant, img_proc)
            if failure_reasons:
                logging.warning('⚠ FAILURE CASE DETECTED for %s:', fname)
                for r in failure_reasons:
                    logging.warning('   → %s', r)
                fail_log = os.path.join(output_dir, 'failure_cases.txt')
                with open(fail_log, 'a') as f:
                    f.write(f'\n{fname}:\n')
                    for r in failure_reasons:
                        f.write(f'  - {r}\n')
            else:
                logging.info('✓ Restoration quality: acceptable')

            # ── Ablation Study ─────────────────────────────────────────────
            if run_ablation:
                logging.info('Running ablation study for %s ...', fname)
                ablation_results = run_ablation_study(
                    img_proc,
                    sat_scale=1.5, nlm_h=6, median_k=3, clahe_clip=1.1,
                    sat_scale_override=1.5, unsharp_amount=0.3,
                    spot_thresh=40, inpaint_radius=2,
                )
                print_ablation_table(ablation_results)
                ablation_out = os.path.join(output_dir, f'ablation_{fname}')
                _save_ablation_grid(ablation_results, ablation_out)
                logging.info('Ablation grid saved to: %s', ablation_out)

            # ── Benchmark vs Classical Methods (#15) ───────────────────────
            if run_benchmark:
                logging.info('Running benchmark comparison for %s ...', fname)
                try:
                    from restoration import run_benchmark_comparison, print_benchmark_table
                    bench_results = run_benchmark_comparison(img_proc)
                    print_benchmark_table(bench_results)
                except Exception:
                    logging.exception('Benchmark failed for %s', fname)

            # ── Save comparison image ──────────────────────────────────────
            if display:
                try:
                    _save_comparison(
                        img, mild_variant, cf, wb_w,
                        os.path.join(output_dir, f'comparison_{fname}'),
                        noise_type=ntype,
                        entropy_val=ent,
                        blur_level=blur_lvl,
                        condition=cond,
                        noise_level=info.get('noise_level'),
                        contrast_score_val=info.get('contrast_score'),
                        mse_val=info.get('mse'),
                        psnr_val=info.get('psnr'),
                        ssim_val=info.get('ssim'),
                        noise_corr=info.get('noise_corr'),
                        sharpness=sharpness,                           # #9
                        adapted_nlm_h=mild_params['nlm_h'],            # #9
                        adapted_clahe=mild_params['clahe_clip'],       # #9
                        adapted_unsharp=mild_params['unsharp_amount'], # #9
                        opt_wb=opt_params['wb_weight'],                # #10
                        opt_sat=opt_params['sat_scale'],               # #10
                        opt_clahe=opt_params['clahe_clip'],            # #10
                        opt_score=opt_score,                           # #10
                        diff_score=diff_score,                         # #11
                        diff_level=diff_level,                         # #11
                        diff_nlm=diff_intensity['nlm_h'],              # #11
                        diff_clahe=diff_intensity['clahe_clip'],       # #11
                        diff_sat=diff_intensity['sat_scale_override'], # #11
                        diff_unsharp=diff_intensity['unsharp_amount'], # #11
                        patch_noise=patch_noise,                       # #12
                        freq_noise=freq_noise,                         # #12
                        combined_noise=combined_noise,                 # #12
                        noise_decision=noise_decision,                 # #12
                    )
                    logging.info('Comparison image saved: comparison_%s', fname)
                except Exception:
                    logging.exception('Error saving comparison for %s', fname)

        except Exception:
            logging.exception('Processing failed for %s', in_path)
            continue


# ── Comparison image ───────────────────────────────────────────────────────────
def _save_comparison(img, mild_variant, cf, wb_w, out_path,
                     noise_type=None, entropy_val=None, blur_level=None,
                     condition=None, noise_level=None, contrast_score_val=None,
                     mse_val=None, psnr_val=None, ssim_val=None, noise_corr=None,
                     sharpness=None, adapted_nlm_h=None,               # #9
                     adapted_clahe=None, adapted_unsharp=None,         # #9
                     opt_wb=None, opt_sat=None,                        # #10
                     opt_clahe=None, opt_score=None,                   # #10
                     diff_score=None, diff_level=None,                 # #11
                     diff_nlm=None, diff_clahe=None,                   # #11
                     diff_sat=None, diff_unsharp=None,                 # #11
                     patch_noise=None, freq_noise=None,                # #12
                     combined_noise=None, noise_decision=None):        # #12
    """Save side-by-side original vs restored comparison image with diagnostics."""

    max_width = 1200
    h, w = img.shape[:2]
    if w > max_width:
        scale           = max_width / w
        img_display     = cv2.resize(img,          (int(w*scale), int(h*scale)),
                                     interpolation=cv2.INTER_AREA)
        variant_display = cv2.resize(mild_variant, (int(w*scale), int(h*scale)),
                                     interpolation=cv2.INTER_AREA)
    else:
        img_display     = img
        variant_display = mild_variant

    orig_rgb = cv2.cvtColor(img_display,     cv2.COLOR_BGR2RGB)
    rest_rgb = cv2.cvtColor(variant_display, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(14, 10), dpi=100)

    axes[0].imshow(orig_rgb)
    axes[0].set_title('Original Image', fontsize=13)
    axes[0].axis('off')

    axes[1].imshow(rest_rgb)
    axes[1].set_title('Restored (Mild)', fontsize=13)
    axes[1].axis('off')

    # ── Existing metrics strings ───────────────────────────────────────────
    detected    = (condition or '').split(' + Noise:')[0] or '-'
    blur_str    = f"{blur_level:.2f}"         if isinstance(blur_level,         (int, float)) else '-'
    ent_str     = f"{entropy_val:.2f}"        if isinstance(entropy_val,        (int, float)) else '-'
    wb_str      = f"{wb_w:.2f}"
    corr_str    = f"{noise_corr:.2f}"         if isinstance(noise_corr,         (int, float)) else '-'
    nlvl_str    = f"{noise_level:.2f}"        if isinstance(noise_level,        (int, float)) else '-'
    cstr        = f"{contrast_score_val:.2f}" if isinstance(contrast_score_val, (int, float)) else '-'
    mse_str     = f"{mse_val:.2f}"            if isinstance(mse_val,            (int, float)) else '-'
    psnr_str    = f"{psnr_val:.2f} dB"        if isinstance(psnr_val,           (int, float)) else '-'
    ssim_str    = f"{ssim_val:.4f}"           if isinstance(ssim_val,           (int, float)) else '-'

    # ── #9 strings ────────────────────────────────────────────────────────
    sharp_str   = f"{sharpness:.3f}"          if isinstance(sharpness,          (int, float)) else '-'
    nlm_str     = str(adapted_nlm_h)          if adapted_nlm_h   is not None                 else '-'
    clahe_str   = f"{adapted_clahe:.2f}"      if isinstance(adapted_clahe,      (int, float)) else '-'
    unsharp_str = f"{adapted_unsharp:.2f}"    if isinstance(adapted_unsharp,    (int, float)) else '-'

    # ── #10 strings ───────────────────────────────────────────────────────
    owb_str    = f"{opt_wb:.2f}"              if isinstance(opt_wb,             (int, float)) else '-'
    osat_str   = f"{opt_sat:.2f}"             if isinstance(opt_sat,            (int, float)) else '-'
    oclahe_str = f"{opt_clahe:.2f}"           if isinstance(opt_clahe,          (int, float)) else '-'
    oscore_str = f"{opt_score:.2f}"           if isinstance(opt_score,          (int, float)) else '-'

    # ── #11 strings ───────────────────────────────────────────────────────
    dscore_str   = f"{diff_score:.2f}"        if isinstance(diff_score,         (int, float)) else '-'
    dlevel_str   = str(diff_level)            if diff_level  is not None                      else '-'
    dnlm_str     = str(diff_nlm)              if diff_nlm    is not None                      else '-'
    dclahe_str   = f"{diff_clahe:.2f}"        if isinstance(diff_clahe,         (int, float)) else '-'
    dsat_str     = f"{diff_sat:.2f}"          if isinstance(diff_sat,           (int, float)) else '-'
    dunsharp_str = f"{diff_unsharp:.2f}"      if isinstance(diff_unsharp,       (int, float)) else '-'

    # ── #12 strings ───────────────────────────────────────────────────────
    pnoise_str  = f"{patch_noise:.2f}"        if isinstance(patch_noise,        (int, float)) else '-'
    fnoise_str  = f"{freq_noise:.2f}"         if isinstance(freq_noise,         (int, float)) else '-'
    cnoise_str  = f"{combined_noise:.2f}"     if isinstance(combined_noise,     (int, float)) else '-'
    ndec_str    = str(noise_decision)         if noise_decision is not None                   else '-'

    # ── Build box text ─────────────────────────────────────────────────────
    box_text = '\n'.join([
        f"Detected Condition : {detected}",
        f"Blur Level         : {blur_str}  (Threshold: 200)",
        f"Image Entropy      : {ent_str}   WB Weight: {wb_str}",
        f"Noise Type         : {noise_type or '-'}   (Corr: {corr_str})",
        f"Noise Level        : {nlvl_str}  ,  Contrast: {cstr}",
        f"MSE: {mse_str}   PSNR: {psnr_str}   SSIM: {ssim_str}",
        f"──────── Continuous Adaptation (#9) ─────────",
        f"Sharpness Score    : {sharp_str}  (0=blurry  \u2192  1=sharp)",
        f"Adapted   nlm_h={nlm_str}   clahe_clip={clahe_str}   unsharp={unsharp_str}",
        f"──────── Optimized Parameters (#10) ──────────",
        f"Best  wb={owb_str}   sat={osat_str}   clahe={oclahe_str}   BRISQUE={oscore_str}",
        f"──────── Difficulty-Aware (#11) ──────────────",
        f"Difficulty Score   : {dscore_str}  \u2192  Level: {dlevel_str}",
        f"Intensity  nlm_h={dnlm_str}   clahe={dclahe_str}   sat={dsat_str}   unsharp={dunsharp_str}",
        f"──────── Noise Estimation (#12) ──────────────",
        f"Patch={pnoise_str}   Freq={fnoise_str}   Combined={cnoise_str}   \u2192 {ndec_str}",
    ])

    plt.suptitle('Image Analysis Details', fontsize=14, y=0.995)
    fig.text(0.5, 0.93, box_text,
             ha='center', va='top', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.6',
                       facecolor='#f5f0d6', edgecolor='black', linewidth=1.0))
    plt.subplots_adjust(top=0.65)
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


# ── Ablation grid ──────────────────────────────────────────────────────────────
def _save_ablation_grid(ablation_results, out_path):
    """Save a grid image showing all ablation variants side by side."""
    order  = ['original', 'full_pipeline', 'no_denoising', 'no_white_balance',
              'no_clahe', 'no_saturation', 'no_unsharp', 'no_fold_suppression']
    labels = {
        'original':            'Original',
        'full_pipeline':       'Full Pipeline',
        'no_denoising':        'No Denoising',
        'no_white_balance':    'No White Balance',
        'no_clahe':            'No CLAHE',
        'no_saturation':       'No Saturation',
        'no_unsharp':          'No Unsharp Mask',
        'no_fold_suppression': 'No Fold Suppression',
    }
    variants = [(k, ablation_results[k]) for k in order if k in ablation_results]
    cols = 4
    rows = (len(variants) + cols - 1) // cols

    max_width = 400
    def downscale(img_v):
        hh, ww = img_v.shape[:2]
        if ww > max_width:
            s = max_width / ww
            return cv2.resize(img_v, (int(ww*s), int(hh*s)), interpolation=cv2.INTER_AREA)
        return img_v

    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4), dpi=80)
    axes = axes.flatten() if rows * cols > 1 else [axes]

    for i, (key, (img_v, b, nq)) in enumerate(variants):
        small = downscale(img_v)
        axes[i].imshow(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        axes[i].set_title(f'{labels.get(key, key)}\nBRISQUE={b:.2f}  NIQE={nq:.2f}',
                          fontsize=9)
        axes[i].axis('off')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle('Ablation Study — Each Step Removed', fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=80, bbox_inches='tight')
    plt.close(fig)


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Batch color restoration for old images')
    parser.add_argument('--input-dir',  '-i', default=None,
                        help='Input folder (default: dataset/old_images)')
    parser.add_argument('--output-dir', '-o', default=None,
                        help='Output folder (default: results/restored_images)')
    parser.add_argument('--no-display', action='store_true',
                        help='Skip saving comparison image')
    parser.add_argument('--ablation',   action='store_true',
                        help='Run ablation study for each image')
    parser.add_argument('--debug',      action='store_true',
                        help='Save debug overlay (green fold lines + red spots)')
    parser.add_argument('--benchmark',  action='store_true',
                        help='Run quality benchmark comparison vs classical methods (#15)')
    parser.add_argument('--file', '-f', default=None,
                        help='Process a single image file (full path)')
    parser.add_argument('--mode',
                        choices=['heuristic', 'difficulty', 'legacy', 'cnn', 'hybrid'],
                        default='heuristic',
                        help='Processing mode: heuristic (default), difficulty (#11 preset), '
                             'legacy (#9+#10), cnn (trained model), hybrid (heuristic+CNN)')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

    base_dir   = os.path.dirname(os.path.abspath(__file__))
    input_dir  = args.input_dir  or os.path.join(base_dir, 'dataset', 'old_images')
    output_dir = args.output_dir or os.path.join(base_dir, 'results', 'restored_images')

    os.makedirs(input_dir,  exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    logging.info('Input folder : %s', input_dir)
    logging.info('Output folder: %s', output_dir)

    process_all(input_dir, output_dir,
                display=(not args.no_display),
                run_ablation=args.ablation,
                debug=args.debug,
                single_file=args.file,
                mode=args.mode,
                run_benchmark=args.benchmark)