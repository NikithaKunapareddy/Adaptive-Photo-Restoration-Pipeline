"""Check saturation levels of images to debug fading detection."""
import cv2
import numpy as np
import os

def check_image_saturation(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f'Cannot read: {img_path}')
        return
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    s_channel = hsv[:, :, 1] / 255.0
    sat_mean = float(np.mean(s_channel))
    sat_min = float(np.min(s_channel))
    sat_max = float(np.max(s_channel))
    
    # Also check contrast
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contrast = float(np.std(gray))
    
    fname = os.path.basename(img_path)
    is_faded = sat_mean < 0.35
    is_low_contrast = contrast < 30.0
    
    print(f'{fname:<25} | Saturation: {sat_mean:.4f} (fade={is_faded}) | Contrast: {contrast:.2f} (low={is_low_contrast})')
    print(f'  └─ Sat range: [{sat_min:.4f}, {sat_max:.4f}]')

print('='*90)
print(f'{"Image":<25} | Saturation (threshold: <0.35) | Contrast (threshold: <30)')
print('='*90)

dataset_dir = 'dataset/old_images'
for fname in sorted(os.listdir(dataset_dir)):
    if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        check_image_saturation(os.path.join(dataset_dir, fname))

print('='*90)
