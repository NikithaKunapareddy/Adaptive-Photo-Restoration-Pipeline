"""
check_saturation.py

Batch analyze colorfulness and adaptive white balance weight for all images.
Helps identify which images are faded and require stronger color restoration.

Output: Table showing colorfulness metric and corresponding WB weight for each image.
"""

import os
import cv2
import numpy as np
from restoration import colorfulness_metric, adaptive_wb_weight_entropy, entropy_metric
import argparse


def analyze_saturation_batch(input_dir):
    """
    Analyze colorfulness and WB weight for all images in directory.
    
    Returns:
        list of dicts with: filename, colorfulness, entropy, wb_weight, status
    """
    results = []
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    
    files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(exts)])
    
    if not files:
        print(f"No images found in {input_dir}")
        return results
    
    print(f"\n{'='*80}")
    print(f"{'File':<30} {'Colorfulness':<16} {'Entropy':<12} {'WB Weight':<12} {'Status':<15}")
    print(f"{'='*80}")
    
    for fname in files:
        in_path = os.path.join(input_dir, fname)
        
        try:
            img = cv2.imread(in_path)
            if img is None:
                print(f"{fname:<30} {'ERROR: Cannot read':<16}")
                continue
            
            # Calculate metrics
            cf = colorfulness_metric(img)
            ent = entropy_metric(img)
            wb_w = adaptive_wb_weight_entropy(ent)
            
            # Determine status based on colorfulness
            if cf < 15:
                status = "VERY FADED"
            elif cf < 33:
                status = "MILDLY FADED"
            elif cf < 45:
                status = "MODERATELY FADED"
            else:
                status = "WELL PRESERVED"
            
            # Print row
            print(f"{fname:<30} {cf:<16.2f} {ent:<12.2f} {wb_w:<12.2f} {status:<15}")
            
            results.append({
                'filename': fname,
                'colorfulness': cf,
                'entropy': ent,
                'wb_weight': wb_w,
                'status': status
            })
        
        except Exception as e:
            print(f"{fname:<30} {'ERROR: ' + str(e)[:15]:<16}")
    
    print(f"{'='*80}\n")
    
    # Print interpretation guide
    print("Interpretation Guide:")
    print("-" * 80)
    print("Colorfulness < 15    → VERY FADED       (WB weight ~0.70) — Needs aggressive restoration")
    print("Colorfulness 15–33   → MILDLY FADED     (WB weight ~0.60) — Moderate fading detected")
    print("Colorfulness 33–45   → MODERATELY FADED (WB weight ~0.45) — Some color loss")
    print("Colorfulness > 45    → WELL PRESERVED   (WB weight ~0.25) — Good color retention")
    print("-" * 80)
    print(f"\nTotal images analyzed: {len(results)}")
    print(f"Faded images (< 33):  {sum(1 for r in results if r['colorfulness'] < 33)}")
    print(f"Preserved images:     {sum(1 for r in results if r['colorfulness'] >= 33)}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Analyze colorfulness and color fade severity for batch of images'
    )
    parser.add_argument('--input-dir', type=str, default='dataset/old_images',
                        help='Input directory containing images (default: dataset/old_images)')
    parser.add_argument('--output', type=str, default=None,
                        help='Optional: save analysis table to CSV file')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory not found: {args.input_dir}")
        return
    
    results = analyze_saturation_batch(args.input_dir)
    
    # Save to CSV if requested
    if args.output and results:
        try:
            import csv
            with open(args.output, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['filename', 'colorfulness', 'entropy', 'wb_weight', 'status'])
                writer.writeheader()
                writer.writerows(results)
            print(f"Results saved to: {args.output}\n")
        except Exception as e:
            print(f"Warning: Could not save CSV: {e}\n")


if __name__ == '__main__':
    main()
