"""
Cylindrical projection for 360-degree panoramas.

For wide-angle (>90°) panoramas, planar homography causes severe distortion.
Cylindrical projection maps images onto a cylinder, which is then "unrolled"
to create a flat panorama. This preserves straight vertical lines.

The key equation:
    x' = f * atan((x - cx) / f)
    y' = f * (y - cy) / sqrt((x - cx)² + f²)

where f is the focal length in pixels.
"""

import numpy as np
from typing import Tuple, List, Optional
from ..core.convolution import gaussian_blur


def cylindrical_warp(image: np.ndarray, 
                     focal_length: float,
                     center: Optional[Tuple[float, float]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Warp an image to cylindrical coordinates.
    
    Projects each pixel onto a cylinder centered at the camera,
    then unrolls the cylinder to get a flat image.
    
    Args:
        image: Input image (H, W) or (H, W, C)
        focal_length: Camera focal length in pixels
        center: Optional (cx, cy) center point. Default is image center.
        
    Returns:
        (warped_image, valid_mask)
    """
    h, w = image.shape[:2]
    
    # Default center
    if center is None:
        cx, cy = w / 2, h / 2
    else:
        cx, cy = center
    
    f = focal_length
    
    # Create output image (same size for simplicity)
    if len(image.shape) == 3:
        output = np.zeros_like(image)
    else:
        output = np.zeros((h, w), dtype=np.float64)
    
    valid_mask = np.zeros((h, w), dtype=bool)
    
    # For each output pixel, find corresponding source pixel
    for y_out in range(h):
        for x_out in range(w):
            # Convert to cylindrical coordinates
            # x_cyl = f * atan((x - cx) / f)
            # y_cyl = f * (y - cy) / sqrt((x - cx)² + f²)
            
            # Inverse mapping: given (x_cyl, y_cyl), find (x, y)
            # x_cyl = f * atan(theta) where theta = (x - cx) / f
            # So theta = tan(x_cyl / f)
            # And x = f * tan(x_cyl / f) + cx
            
            x_cyl = x_out - w / 2  # Center cylindrical coords
            y_cyl = y_out - h / 2
            
            # Inverse cylindrical projection
            theta = x_cyl / f
            
            # Check if theta is valid (|theta| < pi/2 for visible hemisphere)
            if abs(theta) > np.pi / 2:
                continue
            
            x_src = f * np.tan(theta) + cx
            y_src = y_cyl * np.sqrt(1 + np.tan(theta)**2) + cy
            
            # Bilinear interpolation
            if 0 <= x_src < w - 1 and 0 <= y_src < h - 1:
                x0, y0 = int(x_src), int(y_src)
                x1, y1 = x0 + 1, y0 + 1
                
                wx = x_src - x0
                wy = y_src - y0
                
                # Interpolate
                output[y_out, x_out] = (
                    image[y0, x0] * (1 - wx) * (1 - wy) +
                    image[y0, x1] * wx * (1 - wy) +
                    image[y1, x0] * (1 - wx) * wy +
                    image[y1, x1] * wx * wy
                )
                valid_mask[y_out, x_out] = True
    
    return output, valid_mask


def cylindrical_warp_fast(image: np.ndarray,
                          focal_length: float,
                          center: Optional[Tuple[float, float]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fast vectorized cylindrical warp.
    
    Args:
        image: Input image
        focal_length: Focal length in pixels
        center: Optional center point
        
    Returns:
        (warped_image, valid_mask)
    """
    h, w = image.shape[:2]
    f = focal_length
    
    if center is None:
        cx, cy = w / 2, h / 2
    else:
        cx, cy = center
    
    # Create coordinate grids for output
    y_out, x_out = np.mgrid[0:h, 0:w].astype(np.float64)
    
    # Convert to centered cylindrical coordinates
    x_cyl = x_out - w / 2
    y_cyl = y_out - h / 2
    
    # Inverse cylindrical projection
    theta = x_cyl / f
    
    # Only process valid region (|theta| < pi/2)
    valid_theta = np.abs(theta) < (np.pi / 2 - 0.01)  # Small margin
    
    # Compute source coordinates
    tan_theta = np.tan(theta)
    x_src = f * tan_theta + cx
    y_src = y_cyl * np.sqrt(1 + tan_theta**2) + cy
    
    # Valid region mask
    valid_mask = (
        valid_theta &
        (x_src >= 0) & (x_src < w - 1) &
        (y_src >= 0) & (y_src < h - 1)
    )
    
    # Bilinear interpolation
    x0 = np.floor(x_src).astype(np.int32)
    y0 = np.floor(y_src).astype(np.int32)
    
    # Clamp coordinates
    x0 = np.clip(x0, 0, w - 2)
    y0 = np.clip(y0, 0, h - 2)
    x1 = x0 + 1
    y1 = y0 + 1
    
    wx = x_src - x0
    wy = y_src - y0
    
    # Interpolate
    if len(image.shape) == 3:
        output = np.zeros_like(image, dtype=np.float64)
        for c in range(image.shape[2]):
            output[:, :, c] = (
                image[y0, x0, c] * (1 - wx) * (1 - wy) +
                image[y0, x1, c] * wx * (1 - wy) +
                image[y1, x0, c] * (1 - wx) * wy +
                image[y1, x1, c] * wx * wy
            )
    else:
        output = (
            image[y0, x0] * (1 - wx) * (1 - wy) +
            image[y0, x1] * wx * (1 - wy) +
            image[y1, x0] * (1 - wx) * wy +
            image[y1, x1] * wx * wy
        )
    
    # Zero out invalid regions
    if len(image.shape) == 3:
        for c in range(image.shape[2]):
            output[:, :, c] = np.where(valid_mask, output[:, :, c], 0)
    else:
        output = np.where(valid_mask, output, 0)
    
    return output.astype(np.float64), valid_mask


def estimate_focal_length(image_width: int, fov_degrees: float = 60.0) -> float:
    """
    Estimate focal length from field of view.
    
    Args:
        image_width: Image width in pixels
        fov_degrees: Horizontal field of view in degrees
        
    Returns:
        Estimated focal length in pixels
    """
    fov_rad = np.radians(fov_degrees)
    return (image_width / 2) / np.tan(fov_rad / 2)


def compute_cylindrical_shift(kp1_cyl: np.ndarray, 
                               kp2_cyl: np.ndarray) -> Tuple[float, float]:
    """
    Compute translation between two cylindrically-warped images.
    
    For cylindrical panoramas, we only need to estimate (dx, dy) translation,
    not full homography.
    
    Args:
        kp1_cyl: Keypoints from image 1 (Nx2)
        kp2_cyl: Corresponding keypoints from image 2 (Nx2)
        
    Returns:
        (dx, dy) translation
    """
    # Simple median of differences (robust to outliers)
    diff = kp1_cyl - kp2_cyl
    dx = np.median(diff[:, 0])
    dy = np.median(diff[:, 1])
    
    return dx, dy


def ransac_translation(pts1: np.ndarray,
                       pts2: np.ndarray,
                       n_iterations: int = 500,
                       threshold: float = 3.0) -> Tuple[float, float, np.ndarray]:
    """
    Robust translation estimation using RANSAC.
    
    Args:
        pts1: Points from image 1 (Nx2)
        pts2: Corresponding points from image 2 (Nx2)
        n_iterations: Number of RANSAC iterations
        threshold: Inlier threshold in pixels
        
    Returns:
        (dx, dy, inlier_mask)
    """
    n = len(pts1)
    if n < 1:
        return 0.0, 0.0, np.zeros(0, dtype=bool)
    
    best_dx, best_dy = 0.0, 0.0
    best_inliers = 0
    best_mask = np.zeros(n, dtype=bool)
    
    for _ in range(n_iterations):
        # Sample 1 correspondence (translation needs only 1 point)
        idx = np.random.randint(n)
        
        # Compute translation from sample
        dx = pts1[idx, 0] - pts2[idx, 0]
        dy = pts1[idx, 1] - pts2[idx, 1]
        
        # Count inliers
        translated = pts2 + np.array([dx, dy])
        errors = np.sqrt(np.sum((pts1 - translated) ** 2, axis=1))
        inlier_mask = errors < threshold
        n_inliers = np.sum(inlier_mask)
        
        if n_inliers > best_inliers:
            best_inliers = n_inliers
            best_dx, best_dy = dx, dy
            best_mask = inlier_mask.copy()
    
    # Refine using all inliers
    if best_inliers > 0:
        diff = pts1[best_mask] - pts2[best_mask]
        best_dx = np.median(diff[:, 0])
        best_dy = np.median(diff[:, 1])
    
    return best_dx, best_dy, best_mask


from ..blending.multiband import multiband_blend
from ..blending.seam_finding import create_seam_mask

def stitch_cylindrical_pair(img1_cyl: np.ndarray,
                            mask1: np.ndarray,
                            img2_cyl: np.ndarray,
                            mask2: np.ndarray,
                            dx: float,
                            dy: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stitch two cylindrically-warped images using Optimal Seam Finding and Multi-band Blending.
    
    This handles parallax by finding a cut that minimizes difference (Min-Cut),
    then uses multiband blending to seamlessly blend along that cut.
    """
    h1, w1 = img1_cyl.shape[:2]
    h2, w2 = img2_cyl.shape[:2]
    
    # Compute output bounds
    # img1 at (0, 0), img2 at (dx, dy)
    min_x = min(0, dx)
    max_x = max(w1, w2 + dx)
    min_y = min(0, dy)
    max_y = max(h1, h2 + dy)
    
    w_out = int(np.ceil(max_x - min_x))
    h_out = int(np.ceil(max_y - min_y))
    
    offset_x = -min_x
    offset_y = -min_y
    
    # Prepare aligned images
    if len(img1_cyl.shape) == 3:
        shape_out = (h_out, w_out, img1_cyl.shape[2])
    else:
        shape_out = (h_out, w_out)
        
    img1_aligned = np.zeros(shape_out, dtype=np.float64)
    mask1_aligned = np.zeros((h_out, w_out), dtype=bool)
    
    img2_aligned = np.zeros(shape_out, dtype=np.float64)
    mask2_aligned = np.zeros((h_out, w_out), dtype=bool)
    
    # Place img1
    x1 = int(offset_x)
    y1 = int(offset_y)
    img1_aligned[y1:y1+h1, x1:x1+w1] = img1_cyl
    mask1_aligned[y1:y1+h1, x1:x1+w1] = mask1
    
    # Place img2
    x2 = int(offset_x + dx)
    y2 = int(offset_y + dy)
    
    # Clip placement
    place_y_start = max(0, y2)
    place_y_end = min(h_out, y2 + h2)
    place_x_start = max(0, x2)
    place_x_end = min(w_out, x2 + w2)
    
    src_y_start = place_y_start - y2
    src_y_end = src_y_start + (place_y_end - place_y_start)
    src_x_start = place_x_start - x2
    src_x_end = src_x_start + (place_x_end - place_x_start)
    
    if src_y_end > h2: src_y_end = h2 
    if src_x_end > w2: src_x_end = w2
    
    # Verify dimensions
    h_paste = place_y_end - place_y_start
    w_paste = place_x_end - place_x_start
    
    if h_paste > 0 and w_paste > 0:
        img2_aligned[place_y_start:place_y_end, place_x_start:place_x_end] = \
            img2_cyl[src_y_start:src_y_start+h_paste, src_x_start:src_x_start+w_paste]
        mask2_aligned[place_y_start:place_y_end, place_x_start:place_x_end] = \
            mask2[src_y_start:src_y_start+h_paste, src_x_start:src_x_start+w_paste]
            
    # --- Seam Finding Logic ---
    
    # Create the base blend weight mask
    # Start assuming img2 (0) everywhere, then set img1 (1)
    weight_mask = np.zeros((h_out, w_out), dtype=np.float64)
    weight_mask[mask1_aligned] = 1.0
    
    # Identify Overlap
    overlap = mask1_aligned & mask2_aligned
    
    if np.any(overlap):
        # Determine bounding box of overlap to avoid processing full image
        rows, cols = np.where(overlap)
        r_min, r_max = rows.min(), rows.max() + 1
        c_min, c_max = cols.min(), cols.max() + 1
        
        # Extract overlap regions
        crop1 = img1_aligned[r_min:r_max, c_min:c_max]
        crop2 = img2_aligned[r_min:r_max, c_min:c_max]
        
        # Compute optimal seam mask (1 on left of seam, 0 on right)
        seam_mask_crop = create_seam_mask(crop1, crop2)
        
        # Update weight mask in the overlap region
        # Instead of generic 1.0 (from mask1), we use the seam mask
        # Since 'overlap' might be irregular (not a perfect rect), we mask the crop
        # But 'seam_mask_crop' is rectangular.
        # We need to map it back.
        
        # We can fill the weight_mask crop
        # But weight_mask already has 1.0 in this region (from mask1)
        # We need to set it to 0.0 on the RIGHT of the seam.
        # seam_mask_crop is True (1) on Left, False (0) on Right.
        
        current_weights = weight_mask[r_min:r_max, c_min:c_max]
        
        # Where overlap is True in the crop:
        overlap_crop = overlap[r_min:r_max, c_min:c_max]
        
        # Apply seam:
        # If pixel is in overlap, use seam value.
        # Otherwise keep existing (shouldn't happen if we use bounding box of overlap correctly, but overlap might not fill rect)
        
        # Vectorized update
        updated_weights = np.where(overlap_crop, seam_mask_crop.astype(float), current_weights)
        weight_mask[r_min:r_max, c_min:c_max] = updated_weights

    # Use multi-band blending with the smart seam mask
    stitched = multiband_blend(
        img1_aligned, img2_aligned, 
        weight_mask, 
        levels=5
    )
    
    combined_mask = mask1_aligned | mask2_aligned
    
    return stitched, combined_mask


def create_360_panorama(images: List[np.ndarray],
                        focal_lengths: List[float],
                        feature_detector,
                        descriptor_computer,
                        matcher,
                        verbose: bool = True) -> Optional[np.ndarray]:
    """
    Create a 360-degree panorama from multiple images.
    
    Args:
        images: List of images (left to right order)
        focal_lengths: Focal length for each image
        feature_detector: Function to detect features
        descriptor_computer: Function to compute descriptors
        matcher: Function to match features
        verbose: Print progress
        
    Returns:
        360-degree panorama
    """
    n = len(images)
    
    if n == 0:
        return None
    
    if verbose:
        print(f"Creating 360° panorama from {n} images...")
    
    # Step 1: Warp all images to cylindrical coordinates
    if verbose:
        print("Step 1: Cylindrical warping...")
    
    cyl_images = []
    cyl_masks = []
    
    for i, (img, f) in enumerate(zip(images, focal_lengths)):
        if verbose:
            print(f"  Warping image {i+1}/{n}...")
        
        cyl, mask = cylindrical_warp_fast(img, f)
        cyl_images.append(cyl)
        cyl_masks.append(mask)
    
    # Step 2: Detect features in cylindrical images
    if verbose:
        print("Step 2: Detecting features...")
    
    all_keypoints = []
    all_descriptors = []
    
    for i, cyl in enumerate(cyl_images):
        kp, desc = descriptor_computer(cyl)
        all_keypoints.append(kp)
        all_descriptors.append(desc)
        if verbose:
            print(f"  Image {i+1}: {len(kp)} features")
    
    # Step 3: Match consecutive pairs and compute translations
    if verbose:
        print("Step 3: Computing translations...")
    
    translations = []  # (dx, dy) for each adjacent pair
    
    for i in range(n - 1):
        pts1, pts2, matches = matcher(
            all_keypoints[i], all_descriptors[i],
            all_keypoints[i+1], all_descriptors[i+1]
        )
        
        if len(pts1) < 4:
            if verbose:
                print(f"  Pair {i}-{i+1}: Not enough matches!")
            return None
        
        # RANSAC for translation
        dx, dy, inliers = ransac_translation(pts1, pts2)
        n_inliers = np.sum(inliers)
        
        translations.append((dx, dy))
        if verbose:
            print(f"  Pair {i}-{i+1}: dx={dx:.1f}, dy={dy:.1f} ({n_inliers} inliers)")
    
    # Step 4: Stitch all images
    if verbose:
        print("Step 4: Stitching...")
    
    result = cyl_images[0]
    result_mask = cyl_masks[0]
    
    cumulative_dx = 0.0
    cumulative_dy = 0.0
    
    for i in range(n - 1):
        dx, dy = translations[i]
        cumulative_dx += dx
        cumulative_dy += dy
        
        if verbose:
            print(f"  Adding image {i+1}...")
        
        result, result_mask = stitch_cylindrical_pair(
            result, result_mask,
            cyl_images[i+1], cyl_masks[i+1],
            cumulative_dx, cumulative_dy
        )
    
    # Crop to valid region
    valid_rows = np.any(result_mask, axis=1)
    valid_cols = np.any(result_mask, axis=0)
    
    if np.any(valid_rows) and np.any(valid_cols):
        y_min, y_max = np.where(valid_rows)[0][[0, -1]]
        x_min, x_max = np.where(valid_cols)[0][[0, -1]]
        result = result[y_min:y_max+1, x_min:x_max+1]
    
    if verbose:
        print(f"✓ Done! Output size: {result.shape[1]}x{result.shape[0]}")
    
    return result
