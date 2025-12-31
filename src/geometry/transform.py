"""
Perspective transformation and image warping.
Applies homography to transform images.
"""

import numpy as np
from typing import Tuple, Optional
from .homography import apply_homography


def compute_transformed_corners(H: np.ndarray, 
                                  img_shape: Tuple[int, int]) -> np.ndarray:
    """
    Compute where the corners of an image land after homography.
    
    Args:
        H: 3x3 homography matrix
        img_shape: (height, width) of the image
        
    Returns:
        4x2 array of transformed corner coordinates
    """
    h, w = img_shape[:2]
    
    # Four corners (top-left, top-right, bottom-right, bottom-left)
    corners = np.array([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]
    ], dtype=np.float64)
    
    # Transform corners
    transformed = apply_homography(H, corners)
    
    return transformed


def compute_output_bounds(H: np.ndarray, 
                          img_shape: Tuple[int, int],
                          ref_shape: Optional[Tuple[int, int]] = None
                          ) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Compute the output canvas size and offset for warped image.
    
    The output bounds encompass both the warped image and optionally
    a reference image.
    
    Args:
        H: Homography matrix
        img_shape: Shape of image to be warped
        ref_shape: Shape of reference image (optional)
        
    Returns:
        (offset, output_size)
        - offset: (x_offset, y_offset) translation to apply
        - output_size: (height, width) of output canvas
    """
    # Get transformed corners
    corners = compute_transformed_corners(H, img_shape)
    
    # Find bounding box of transformed corners
    min_x = corners[:, 0].min()
    max_x = corners[:, 0].max()
    min_y = corners[:, 1].min()
    max_y = corners[:, 1].max()
    
    # Include reference image if provided
    if ref_shape is not None:
        ref_h, ref_w = ref_shape[:2]
        min_x = min(min_x, 0)
        max_x = max(max_x, ref_w - 1)
        min_y = min(min_y, 0)
        max_y = max(max_y, ref_h - 1)
    
    # Compute offset (translation to make coordinates positive)
    offset = np.array([-min_x, -min_y])
    
    # Compute output size
    output_w = int(np.ceil(max_x - min_x)) + 1
    output_h = int(np.ceil(max_y - min_y)) + 1
    
    return offset, (output_h, output_w)


def bilinear_interpolate(image: np.ndarray, x: float, y: float) -> np.ndarray:
    """
    Sample image at non-integer location using bilinear interpolation.
    
    Args:
        image: Input image (H, W) or (H, W, C)
        x: X coordinate (can be non-integer)
        y: Y coordinate (can be non-integer)
        
    Returns:
        Interpolated pixel value
    """
    h, w = image.shape[:2]
    
    # Get integer coordinates
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = x0 + 1
    y1 = y0 + 1
    
    # Check bounds
    if x0 < 0 or x1 >= w or y0 < 0 or y1 >= h:
        if len(image.shape) == 3:
            return np.zeros(image.shape[2])
        return 0.0
    
    # Interpolation weights
    wx = x - x0
    wy = y - y0
    
    # Get four neighbors
    p00 = image[y0, x0]
    p01 = image[y0, x1]
    p10 = image[y1, x0]
    p11 = image[y1, x1]
    
    # Bilinear interpolation
    result = (p00 * (1 - wx) * (1 - wy) +
              p01 * wx * (1 - wy) +
              p10 * (1 - wx) * wy +
              p11 * wx * wy)
    
    return result


def warp_perspective(image: np.ndarray,
                     H: np.ndarray,
                     output_shape: Tuple[int, int],
                     offset: np.ndarray = None,
                     border_value: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Warp image using homography with inverse mapping.
    
    For each pixel in the output, we compute its source location
    using H^-1, then sample the source image using bilinear interpolation.
    
    Args:
        image: Input image to warp
        H: 3x3 homography matrix (maps from source to destination)
        output_shape: (height, width) of output image
        offset: (x, y) offset to apply in destination space
        border_value: Value for out-of-bounds pixels
        
    Returns:
        (warped_image, valid_mask)
        - warped_image: The transformed image
        - valid_mask: Binary mask of valid (non-border) pixels
    """
    h_out, w_out = output_shape[:2]
    
    # Default offset
    if offset is None:
        offset = np.array([0.0, 0.0])
    
    # Create output array
    if len(image.shape) == 3:
        output = np.full((h_out, w_out, image.shape[2]), border_value, dtype=np.float64)
    else:
        output = np.full((h_out, w_out), border_value, dtype=np.float64)
    
    valid_mask = np.zeros((h_out, w_out), dtype=bool)
    
    # Compute inverse homography for backward mapping
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return output, valid_mask
    
    # Account for offset: we want H_inv that maps destination coords to source
    # If offset is applied, destination coords are shifted
    # dst = H @ src + offset  =>  src = H_inv @ (dst - offset)
    # Create adjusted inverse: H_inv_adj = H_inv @ T(-offset)
    T_offset = np.array([
        [1, 0, -offset[0]],
        [0, 1, -offset[1]],
        [0, 0, 1]
    ], dtype=np.float64)
    
    H_inv_adj = H_inv @ T_offset
    
    # For each output pixel, find source location
    src_h, src_w = image.shape[:2]
    
    for y_out in range(h_out):
        for x_out in range(w_out):
            # Transform to source coordinates
            dst_pt = np.array([x_out, y_out, 1.0])
            src_pt_h = H_inv_adj @ dst_pt
            
            # Perspective division
            if abs(src_pt_h[2]) < 1e-10:
                continue
            
            x_src = src_pt_h[0] / src_pt_h[2]
            y_src = src_pt_h[1] / src_pt_h[2]
            
            # Check if source location is valid
            if 0 <= x_src < src_w - 1 and 0 <= y_src < src_h - 1:
                output[y_out, x_out] = bilinear_interpolate(image, x_src, y_src)
                valid_mask[y_out, x_out] = True
    
    return output, valid_mask


def warp_perspective_fast(image: np.ndarray,
                          H: np.ndarray,
                          output_shape: Tuple[int, int],
                          offset: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fast perspective warp using vectorized operations.
    
    This is much faster than the per-pixel version for large images.
    
    Args:
        image: Input image
        H: Homography matrix
        output_shape: Output size (height, width)
        offset: Translation offset
        
    Returns:
        (warped_image, valid_mask)
    """
    h_out, w_out = output_shape[:2]
    src_h, src_w = image.shape[:2]
    
    if offset is None:
        offset = np.array([0.0, 0.0])
    
    # Compute inverse homography
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        if len(image.shape) == 3:
            return np.zeros((h_out, w_out, image.shape[2])), np.zeros((h_out, w_out), dtype=bool)
        return np.zeros((h_out, w_out)), np.zeros((h_out, w_out), dtype=bool)
    
    # Create coordinate grid for output
    y_coords, x_coords = np.mgrid[0:h_out, 0:w_out]
    
    # Apply offset
    x_coords = x_coords.astype(np.float64) - offset[0]
    y_coords = y_coords.astype(np.float64) - offset[1]
    
    # Homogeneous coordinates
    ones = np.ones_like(x_coords)
    
    # Transform all points at once
    # Stack into 3 x (h_out * w_out)
    coords = np.stack([x_coords.ravel(), y_coords.ravel(), ones.ravel()], axis=0)
    
    # Apply inverse homography
    src_coords_h = H_inv @ coords  # 3 x N
    
    # Perspective division
    w = src_coords_h[2, :]
    w = np.where(np.abs(w) < 1e-10, 1e-10, w)
    
    x_src = (src_coords_h[0, :] / w).reshape(h_out, w_out)
    y_src = (src_coords_h[1, :] / w).reshape(h_out, w_out)
    
    # Valid mask
    valid_mask = ((x_src >= 0) & (x_src < src_w - 1) & 
                  (y_src >= 0) & (y_src < src_h - 1))
    
    # Integer and fractional parts for bilinear interpolation
    x0 = np.floor(x_src).astype(np.int32)
    y0 = np.floor(y_src).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1
    
    # Clamp to valid range
    x0 = np.clip(x0, 0, src_w - 1)
    x1 = np.clip(x1, 0, src_w - 1)
    y0 = np.clip(y0, 0, src_h - 1)
    y1 = np.clip(y1, 0, src_h - 1)
    
    # Interpolation weights
    wx = x_src - np.floor(x_src)
    wy = y_src - np.floor(y_src)
    
    # Bilinear interpolation
    if len(image.shape) == 3:
        output = np.zeros((h_out, w_out, image.shape[2]), dtype=np.float64)
        for c in range(image.shape[2]):
            p00 = image[y0, x0, c]
            p01 = image[y0, x1, c]
            p10 = image[y1, x0, c]
            p11 = image[y1, x1, c]
            
            output[:, :, c] = (p00 * (1 - wx) * (1 - wy) +
                               p01 * wx * (1 - wy) +
                               p10 * (1 - wx) * wy +
                               p11 * wx * wy)
    else:
        p00 = image[y0, x0]
        p01 = image[y0, x1]
        p10 = image[y1, x0]
        p11 = image[y1, x1]
        
        output = (p00 * (1 - wx) * (1 - wy) +
                  p01 * wx * (1 - wy) +
                  p10 * (1 - wx) * wy +
                  p11 * wx * wy)
    
    # Set invalid pixels to 0
    if len(image.shape) == 3:
        output[~valid_mask] = 0
    else:
        output[~valid_mask] = 0
    
    return output, valid_mask


def stitch_images(img1: np.ndarray,
                  img2: np.ndarray,
                  H: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Stitch two images together using homography.
    
    Image 1 is the reference (stays in place).
    Image 2 is warped to align with image 1.
    
    Args:
        img1: Reference image
        img2: Image to warp
        H: Homography that maps img2 -> img1 coordinate system
        
    Returns:
        (stitched, mask1, mask2)
        - stitched: Combined image
        - mask1: Validity mask for img1
        - mask2: Validity mask for warped img2
    """
    # Compute output bounds
    offset, output_shape = compute_output_bounds(H, img2.shape[:2], img1.shape[:2])
    
    # Create output canvas
    h_out, w_out = output_shape
    if len(img1.shape) == 3:
        canvas = np.zeros((h_out, w_out, img1.shape[2]), dtype=np.float64)
    else:
        canvas = np.zeros((h_out, w_out), dtype=np.float64)
    
    mask1 = np.zeros((h_out, w_out), dtype=bool)
    mask2 = np.zeros((h_out, w_out), dtype=bool)
    
    # Place img1 on canvas (with offset)
    x_off = int(offset[0])
    y_off = int(offset[1])
    
    h1, w1 = img1.shape[:2]
    y1_start = max(0, y_off)
    y1_end = min(h_out, y_off + h1)
    x1_start = max(0, x_off)
    x1_end = min(w_out, x_off + w1)
    
    # Corresponding source coordinates
    src_y_start = max(0, -y_off)
    src_x_start = max(0, -x_off)
    src_y_end = src_y_start + (y1_end - y1_start)
    src_x_end = src_x_start + (x1_end - x1_start)
    
    canvas[y1_start:y1_end, x1_start:x1_end] = img1[src_y_start:src_y_end, src_x_start:src_x_end]
    mask1[y1_start:y1_end, x1_start:x1_end] = True
    
    # Warp img2 onto canvas
    warped, valid = warp_perspective_fast(img2, H, output_shape, offset)
    mask2 = valid
    
    return canvas, mask1, mask2, warped
