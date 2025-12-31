# 360° Panorama Stitcher

A complete panorama stitching system built **from scratch** using only NumPy and PIL.  
No OpenCV, no scipy — just pure Python implementations of computer vision algorithms.

---

## Quick Start

```bash
# Install dependencies
pip install numpy pillow

# Run demo
python main.py

# Stitch 360° panorama from a dataset
python main.py --360 grail

# Stitch your own images
python main.py --images image1.jpg image2.jpg image3.jpg -o output/my_panorama.png
```

---

## Usage

### Command Line Interface

```bash
python main.py [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--360 DATASET` | Use preset dataset (`grail`, `parrington`, `library`, `xue`) |
| `--folder PATH` | **Your own folder** of 360° images |
| `-f, --focal N` | Focal length in pixels (default: 700) |
| `--reverse` | Reverse image order (for counter-clockwise capture) |
| `--images PATH...` | Stitch individual image files (planar mode) |
| `-o, --output PATH` | Output file path |
| `--demo` | Run demo with synthetic images |

### Examples

```bash
# Demo with synthetic images
python main.py

# 360° from preset dataset
python main.py --360 grail

# 360° from YOUR OWN folder
python main.py --folder path/to/my_images/ -f 700

# Counter-clockwise captured images
python main.py --folder my_images/ -f 800 --reverse

# Stitch 3 individual images (planar)
python main.py --images left.jpg center.jpg right.jpg

# Custom output path
python main.py --folder photos/ -f 750 -o result.png
```

### Focal Length Tips

The focal length depends on your camera. Common values:

| Camera | Approximate Focal Length |
|--------|-------------------------|
| Phone (wide) | 600-800 px |
| Phone (normal) | 800-1000 px |
| DSLR (18mm) | 600-700 px |
| DSLR (35mm) | 1000-1200 px |

If unsure, start with `700` and adjust if you see vertical drift.

---

## Datasets

Pre-configured 360° datasets in `data/`:

| Dataset | Images | Resolution | Scene |
|---------|--------|------------|-------|
| `grail` | 18 | 640×480 | Indoor |
| `parrington` | 18 | 640×480 | Outdoor |
| `library` | 14 | 640×480 | Outdoor |
| `xue` | 16 | 1024×768 | Outdoor |

> **Credit:** The original image sequences is from [SSARCandy/panoramas-image-stitching](https://github.com/SSARCandy/panoramas-image-stitching/)

---

## Project Structure

```
PanoramaStitcher/
├── main.py              # Entry point
├── src/
│   ├── core/            # Image I/O, convolution, math
│   ├── features/        # Harris corners, descriptors, matching
│   ├── geometry/        # Homography, RANSAC, cylindrical projection
│   ├── blending/        # Alpha blend, multi-band, seam finding
│   └── pipeline/        # PanoramaStitcher class
├── data/                # Test datasets
├── output/              # Generated panoramas
├── THESIS_REPORT.tex    # Full technical report (LaTeX)
└── requirements.txt
```

---

## Algorithms Implemented

| Component | Algorithm |
|-----------|-----------|
| Feature Detection | Harris Corner Detector |
| Feature Description | BRIEF-like Binary Descriptors (256 bits) |
| Feature Matching | Hamming Distance + Lowe's Ratio Test |
| Robust Estimation | RANSAC (Homography & Translation) |
| Projection | Cylindrical Warping for 360° |
| Blending | Multi-band Laplacian Pyramid |
| Seam Optimization | Dynamic Programming Min-Cut |

---

## Requirements

- Python 3.7+
- NumPy
- Pillow (PIL)

```bash
pip install -r requirements.txt
```

---

## Example Output

```
$ python main.py --360 grail

============================================================
360° CYLINDRICAL PANORAMA: grail
============================================================
Path: data/grail
Focal Length: 627.0 px
Images: 18

Cylindrical warping...
Feature detection...
  Image 1: 520 features
  Image 2: 485 features
  ...
Stitching...
  Pair 0-1: dx=285.0, dy=2.1 (72/95 inliers)
  Pair 1-2: dx=290.0, dy=1.5 (68/88 inliers)
  ...

✓ 360° Panorama saved to output/grail_360_panorama.png
  Size: 4082 x 880
```

---

## License

Educational project for Computer Vision coursework.
