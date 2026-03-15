# Color Restoration of Old/Damaged Images

This project restores faded or degraded colors in old photographs using Digital Image Processing techniques (OpenCV + NumPy).

Requirements
- Python 3.8+ (or any installed Python that provides `py` or `python` on PATH)
- pip
- Required packages (install with pip):

```bash
pip install -r requirements.txt
```

Or individually:

```bash
pip install opencv-python numpy matplotlib
```

Usage
1. Place your input images (jpg/png/bmp/tiff) into the folder:

- color_restoration_project/dataset/old_images/

2. Run the program from the project folder. If `python` is on your PATH:

```powershell
cd C:/Users/nikit/Desktop/dip/color_restoration_project
python main.py
```

If `python` is not available but the `py` launcher is installed on Windows, run:

```powershell
cd C:/Users/nikit/Desktop/dip/color_restoration_project
py main.py
```

3. The script will display Original vs Restored images and save restored files to:

- color_restoration_project/results/restored_images/

Notes
- The pipeline applies the following steps to each input image:
  1. Bilateral filtering for noise removal (edge preserving)
  2. Gray World white balance in LAB to correct color tints
  3. CLAHE on L channel for contrast enhancement
  4. Saturation boost in HSV to restore faded colors
  5. Sharpening via a simple kernel

- Designed to run locally in VS Code without web frameworks.

If you want, I can try running `py main.py` now (attempt to use Windows launcher). Say "run" to proceed.
