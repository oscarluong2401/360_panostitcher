"""
Panorama Stitcher - Main Entry Point

Usage:
    python main.py                    # Run demo with synthetic images
    python main.py --360 grail        # Run 360° panorama on dataset
    python main.py --images img1.jpg img2.jpg img3.jpg

Built from scratch using only NumPy and PIL.
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import numpy as np
from src.core.image import Image
from src.pipeline.stitcher import PanoramaStitcher


# ============================================================================
# Configuration
# ============================================================================

DATASETS = {
    'grail': {
        'path': 'data/grail',
        'focal_length': 627.0,
        'reverse': True,  # Counter-clockwise
    },
    'parrington': {
        'path': 'data/parrington',
        'focal_length': 705.0,
        'reverse': True,
    },
    'library': {
        'path': 'data/library',
        'focal_length': 659.0,
        'reverse': False,  # Clockwise
    },
    'xue': {
        'path': 'data/Xue-Mountain-Enterance',
        'focal_length': 830.0,
        'reverse': False,
    },
}


# ============================================================================
# Stitching Functions
# ============================================================================

def stitch_planar(image_paths: list, output_path: str = 'output/panorama.png'):
    """
    Stitch images using homography-based planar stitching.
    Best for <90° field of view.
    """
    print("=" * 60)
    print("PLANAR PANORAMA STITCHING")
    print("=" * 60)
    
    stitcher = PanoramaStitcher(
        harris_threshold=0.01,
        max_keypoints=800,
        ratio_threshold=0.75,
        ransac_iterations=2000,
        ransac_threshold=5.0,
        blend_mode='multiband',
        verbose=True
    )
    
    result = stitcher.stitch_from_paths(image_paths)
    
    if result is not None:
        Path(output_path).parent.mkdir(exist_ok=True)
        Image(result).save(output_path)
        print(f"\n✓ Panorama saved to {output_path}")
        return True
    else:
        print("\n✗ Stitching failed!")
        return False


def stitch_360(dataset_name: str, output_path: str = None):
    """
    Stitch 360° panorama using cylindrical projection.
    """
    from src.geometry.cylindrical import (
        cylindrical_warp_fast, 
        ransac_translation, 
        stitch_cylindrical_pair
    )
    from src.features.harris import harris_corners
    from src.features.nms import non_maximum_suppression
    from src.features.descriptor import compute_descriptors
    from src.features.matching import match_features, get_matched_points
    
    if dataset_name not in DATASETS:
        print(f"Unknown dataset: {dataset_name}")
        print(f"Available: {list(DATASETS.keys())}")
        return False
    
    config = DATASETS[dataset_name]
    dataset_path = Path(__file__).parent / config['path']
    focal_length = config['focal_length']
    
    print("=" * 60)
    print(f"360° CYLINDRICAL PANORAMA: {dataset_name}")
    print("=" * 60)
    print(f"Path: {dataset_path}")
    print(f"Focal Length: {focal_length} px")
    
    # Load images
    image_files = sorted(dataset_path.glob('*.jpg'))
    if config['reverse']:
        image_files = list(reversed(image_files))
        print("(Reversed for counter-clockwise dataset)")
    
    print(f"Images: {len(image_files)}")
    
    images = []
    for f in image_files:
        img = Image.load(f)
        images.append(img.data)
    
    # Warp to cylindrical
    print("\nCylindrical warping...")
    cyl_images = []
    masks = []
    for i, img in enumerate(images):
        cyl, mask = cylindrical_warp_fast(img, focal_length)
        cyl_images.append(cyl)
        masks.append(mask)
        print(f"  Image {i+1}: {img.shape[:2]} -> cylindrical")
    
    # Detect features
    print("\nFeature detection...")
    all_kp = []
    all_desc = []
    for i, cyl in enumerate(cyl_images):
        gray = cyl[:,:,0]*0.299 + cyl[:,:,1]*0.587 + cyl[:,:,2]*0.114 if len(cyl.shape)==3 else cyl
        corners = harris_corners(gray, threshold=0.005)
        corners = non_maximum_suppression(corners, radius=10, max_keypoints=800)
        kp, desc = compute_descriptors(gray, corners)
        all_kp.append(kp)
        all_desc.append(desc)
        print(f"  Image {i+1}: {len(kp)} features")
    
    # Progressive stitching
    print("\nStitching...")
    result = cyl_images[0]
    result_mask = masks[0]
    
    for i in range(len(cyl_images) - 1):
        # Match features - returns (matched_kp1, matched_kp2, matches)
        matched_kp1, matched_kp2, matches = match_features(
            all_kp[i], all_desc[i], all_kp[i+1], all_desc[i+1]
        )
        
        # Convert to point arrays
        pts1 = np.array([[kp.x, kp.y] for kp in matched_kp1])
        pts2 = np.array([[kp.x, kp.y] for kp in matched_kp2])
        
        if len(pts1) < 10:
            print(f"  Pair {i}-{i+1}: Not enough matches ({len(pts1)})")
            continue
        
        # Estimate translation
        dx, dy, inliers = ransac_translation(pts1, pts2, threshold=3.0)
        n_inliers = np.sum(inliers)
        print(f"  Pair {i}-{i+1}: dx={dx:.1f}, dy={dy:.1f} ({n_inliers}/{len(pts1)} inliers)")
        
        # Stitch with seam finding + multiband blending
        result, result_mask = stitch_cylindrical_pair(
            result, result_mask,
            cyl_images[i+1], masks[i+1],
            dx, dy
        )
    
    # Crop valid region
    valid_rows = np.any(result_mask, axis=1)
    valid_cols = np.any(result_mask, axis=0)
    r_min, r_max = np.where(valid_rows)[0][[0, -1]]
    c_min, c_max = np.where(valid_cols)[0][[0, -1]]
    result = result[r_min:r_max+1, c_min:c_max+1]
    
    # Save
    if output_path is None:
        output_path = f'output/{dataset_name}_360_panorama.png'
    
    Path(output_path).parent.mkdir(exist_ok=True)
    Image(result).save(output_path)
    
    print(f"\n✓ 360° Panorama saved to {output_path}")
    print(f"  Size: {result.shape[1]} x {result.shape[0]}")
    return True


def demo_synthetic():
    """Quick demo with synthetic checkerboard images."""
    print("=" * 60)
    print("DEMO: Synthetic Image Stitching")
    print("=" * 60)
    
    # Create checkerboard images
    h, w = 400, 600
    cell = 40
    overlap = 200
    
    def checkerboard(width, offset=0):
        img = np.zeros((h, width, 3), dtype=np.float64)
        for y in range(h):
            for x in range(width):
                if ((y // cell) + ((x + offset) // cell)) % 2 == 0:
                    img[y, x] = [0.9, 0.9, 0.9]
                else:
                    img[y, x] = [0.3, 0.3, 0.3]
        return img
    
    img1 = checkerboard(w, 0)
    img2 = checkerboard(w, w - overlap)
    
    # Add markers
    for y, x, c in [(100, 100, [1,0,0]), (200, 300, [0,1,0]), (300, 500, [0,0,1])]:
        if x < w: img1[y-10:y+10, x-10:x+10] = c
        x2 = x - (w - overlap)
        if 0 <= x2 < w: img2[y-10:y+10, max(0,x2-10):min(w,x2+10)] = c
    
    # Save inputs
    Path('output').mkdir(exist_ok=True)
    Image(img1).save('output/demo_input1.png')
    Image(img2).save('output/demo_input2.png')
    
    # Stitch
    stitcher = PanoramaStitcher(harris_threshold=0.005, verbose=True)
    result = stitcher.stitch_pair(img1, img2)
    
    if result is not None:
        Image(result).save('output/demo_panorama.png')
        print("\n✓ Demo complete! Check output/ directory")
    else:
        print("\n✗ Demo failed")


# ============================================================================
# Main
# ============================================================================

def stitch_360_folder(folder_path: str, focal_length: float, output_path: str = None, reverse: bool = False):
    """
    Stitch 360° panorama from a custom folder of images.
    """
    from src.geometry.cylindrical import (
        cylindrical_warp_fast, 
        ransac_translation, 
        stitch_cylindrical_pair
    )
    from src.features.harris import harris_corners
    from src.features.nms import non_maximum_suppression
    from src.features.descriptor import compute_descriptors
    from src.features.matching import match_features, get_matched_points
    
    dataset_path = Path(folder_path)
    if not dataset_path.exists():
        print(f"Error: Folder not found: {folder_path}")
        return False
    
    print("=" * 60)
    print(f"360° CYLINDRICAL PANORAMA: {dataset_path.name}")
    print("=" * 60)
    print(f"Path: {dataset_path}")
    print(f"Focal Length: {focal_length} px")
    
    # Load images (jpg, png, jpeg)
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_files.extend(dataset_path.glob(ext))
    image_files = sorted(image_files)
    
    if len(image_files) < 2:
        print(f"Error: Need at least 2 images, found {len(image_files)}")
        return False
    
    if reverse:
        image_files = list(reversed(image_files))
        print("(Reversed image order)")
    
    print(f"Images: {len(image_files)}")
    for f in image_files[:5]:
        print(f"  - {f.name}")
    if len(image_files) > 5:
        print(f"  ... and {len(image_files) - 5} more")
    
    images = []
    for f in image_files:
        img = Image.load(f)
        images.append(img.data)
    
    # Warp to cylindrical
    print("\nCylindrical warping...")
    cyl_images = []
    masks = []
    for i, img in enumerate(images):
        cyl, mask = cylindrical_warp_fast(img, focal_length)
        cyl_images.append(cyl)
        masks.append(mask)
    print(f"  Warped {len(images)} images")
    
    # Detect features
    print("\nFeature detection...")
    all_kp = []
    all_desc = []
    for i, cyl in enumerate(cyl_images):
        gray = cyl[:,:,0]*0.299 + cyl[:,:,1]*0.587 + cyl[:,:,2]*0.114 if len(cyl.shape)==3 else cyl
        corners = harris_corners(gray, threshold=0.005)
        corners = non_maximum_suppression(corners, radius=10, max_keypoints=800)
        kp, desc = compute_descriptors(gray, corners)
        all_kp.append(kp)
        all_desc.append(desc)
    print(f"  Detected features in {len(images)} images")
    
    # Progressive stitching
    print("\nStitching...")
    result = cyl_images[0]
    result_mask = masks[0]
    
    for i in range(len(cyl_images) - 1):
        matched_kp1, matched_kp2, matches = match_features(
            all_kp[i], all_desc[i], all_kp[i+1], all_desc[i+1]
        )
        pts1 = np.array([[kp.x, kp.y] for kp in matched_kp1])
        pts2 = np.array([[kp.x, kp.y] for kp in matched_kp2])
        
        if len(pts1) < 10:
            print(f"  Pair {i}-{i+1}: Not enough matches ({len(pts1)}), skipping")
            continue
        
        dx, dy, inliers = ransac_translation(pts1, pts2, threshold=3.0)
        n_inliers = np.sum(inliers)
        print(f"  Pair {i}-{i+1}: dx={dx:.1f}, dy={dy:.1f} ({n_inliers}/{len(pts1)} inliers)")
        
        result, result_mask = stitch_cylindrical_pair(
            result, result_mask,
            cyl_images[i+1], masks[i+1],
            dx, dy
        )
    
    # Crop valid region
    valid_rows = np.any(result_mask, axis=1)
    valid_cols = np.any(result_mask, axis=0)
    r_min, r_max = np.where(valid_rows)[0][[0, -1]]
    c_min, c_max = np.where(valid_cols)[0][[0, -1]]
    result = result[r_min:r_max+1, c_min:c_max+1]
    
    # Save
    if output_path is None:
        output_path = f'output/{dataset_path.name}_360_panorama.png'
    
    Path(output_path).parent.mkdir(exist_ok=True)
    Image(result).save(output_path)
    
    print(f"\n✓ 360° Panorama saved to {output_path}")
    print(f"  Size: {result.shape[1]} x {result.shape[0]}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Panorama Stitcher - Built from scratch without OpenCV',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                              # Run demo
  python main.py --360 grail                  # Use preset dataset
  python main.py --folder my_images/ -f 700   # Your own 360° folder
  python main.py --images a.jpg b.jpg c.jpg   # Stitch individual files
        """
    )
    parser.add_argument('--360', dest='dataset_360', metavar='DATASET',
                        help='Create 360° panorama from preset (grail/parrington/library/xue)')
    parser.add_argument('--folder', metavar='PATH',
                        help='Create 360° panorama from your own image folder')
    parser.add_argument('-f', '--focal', type=float, default=700.0,
                        help='Focal length in pixels for --folder (default: 700)')
    parser.add_argument('--reverse', action='store_true',
                        help='Reverse image order (for counter-clockwise capture)')
    parser.add_argument('--images', nargs='+', metavar='PATH',
                        help='Stitch these image files (planar homography)')
    parser.add_argument('-o', '--output', default=None,
                        help='Output file path')
    parser.add_argument('--demo', action='store_true',
                        help='Run demo with synthetic images')
    
    args = parser.parse_args()
    
    if args.dataset_360:
        stitch_360(args.dataset_360, args.output)
    elif args.folder:
        stitch_360_folder(args.folder, args.focal, args.output, args.reverse)
    elif args.images:
        output = args.output or 'output/panorama.png'
        stitch_planar(args.images, output)
    else:
        # Default: run demo
        demo_synthetic()


if __name__ == '__main__':
    main()
