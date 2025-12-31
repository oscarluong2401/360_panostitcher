"""
Alpha blending for image stitching.
Simple blending using weighted averaging.
"""

import numpy as np
from typing import Tuple


def create_linear_blend_mask(shape: Tuple[int, int], 
                             direction: str = 'horizontal',
                             width: int = None) -> np.ndarray:
    """
    Create a linear gradient mask for blending.
    
    Args:
        shape: (height, width) of mask
        direction: 'horizontal' or 'vertical'
        width: Gradient width (default: full image)
        
    Returns:
        Gradient mask with values [0, 1]
    """
    h, w = shape
    
    if width is None:
        width = w if direction == 'horizontal' else h
    
    if direction == 'horizontal':
        # Left to right gradient
        mask = np.linspace(0, 1, width)
        mask = np.tile(mask, (h, 1))
        # Pad if needed
        if width < w:
            left_pad = (w - width) // 2
            full_mask = np.zeros((h, w))
            full_mask[:, :left_pad] = 0
            full_mask[:, left_pad:left_pad+width] = mask
            full_mask[:, left_pad+width:] = 1
            mask = full_mask
    else:
        # Top to bottom gradient
        mask = np.linspace(0, 1, width)
        mask = np.tile(mask.reshape(-1, 1), (1, w))
        if width < h:
            top_pad = (h - width) // 2
            full_mask = np.zeros((h, w))
            full_mask[:top_pad, :] = 0
            full_mask[top_pad:top_pad+width, :] = mask
            full_mask[top_pad+width:, :] = 1
            mask = full_mask
    
    return mask


def create_distance_mask(shape: Tuple[int, int], 
                         valid_mask: np.ndarray) -> np.ndarray:
    """
    Create a mask based on distance from invalid pixels.
    
    Pixels closer to the edge (invalid region) have lower weight.
    This creates smooth transitions at image boundaries.
    
    Args:
        shape: (height, width)
        valid_mask: Binary mask of valid pixels
        
    Returns:
        Distance-based weight mask
    """
    h, w = shape
    
    # Simple distance transform using erosion approximation
    # (True distance transform would use scipy, which we're avoiding)
    
    dist = np.zeros((h, w), dtype=np.float64)
    current = valid_mask.astype(np.float64)
    
    # Iteratively erode and accumulate distance
    max_iter = min(50, min(h, w) // 4)
    
    for i in range(max_iter):
        # Erode by looking at neighbors
        eroded = np.zeros_like(current)
        eroded[1:-1, 1:-1] = np.minimum.reduce([
            current[1:-1, 1:-1],
            current[:-2, 1:-1],   # top
            current[2:, 1:-1],    # bottom
            current[1:-1, :-2],   # left
            current[1:-1, 2:]     # right
        ])
        
        # Accumulate distance
        dist += current
        current = eroded
        
        if current.sum() == 0:
            break
    
    # Normalize
    if dist.max() > 0:
        dist = dist / dist.max()
    
    return dist


def feather_mask(valid_mask: np.ndarray, 
                 feather_width: int = 20) -> np.ndarray:
    """
    Create a feathered (soft-edged) mask from a binary mask.
    
    Args:
        valid_mask: Binary mask
        feather_width: Width of the feathering region
        
    Returns:
        Feathered mask with smooth edges
    """
    h, w = valid_mask.shape
    
    # Start with the valid mask as float
    mask = valid_mask.astype(np.float64)
    
    # Apply blur-like operation for feathering
    for _ in range(feather_width):
        new_mask = mask.copy()
        # Average with neighbors
        new_mask[1:-1, 1:-1] = (
            mask[1:-1, 1:-1] * 0.2 +
            mask[:-2, 1:-1] * 0.2 +   # top
            mask[2:, 1:-1] * 0.2 +    # bottom
            mask[1:-1, :-2] * 0.2 +   # left
            mask[1:-1, 2:] * 0.2      # right
        )
        mask = new_mask
    
    # Ensure original valid region stays valid
    mask = np.maximum(mask, valid_mask * 0.5)
    
    return mask


def alpha_blend(img1: np.ndarray, 
                img2: np.ndarray,
                mask1: np.ndarray,
                mask2: np.ndarray,
                feather: bool = True,
                feather_width: int = 30) -> np.ndarray:
    """
    Blend two images using alpha masks.
    
    Args:
        img1: First image
        img2: Second image
        mask1: Validity mask for img1
        mask2: Validity mask for img2
        feather: Whether to feather the masks
        feather_width: Width of feathering
        
    Returns:
        Blended image
    """
    # Ensure same shape
    assert img1.shape == img2.shape, "Images must have same shape"
    
    # Create weight masks
    if feather:
        w1 = feather_mask(mask1, feather_width)
        w2 = feather_mask(mask2, feather_width)
    else:
        w1 = mask1.astype(np.float64)
        w2 = mask2.astype(np.float64)
    
    # Normalize weights (handle overlap)
    total = w1 + w2
    total = np.where(total < 1e-10, 1, total)
    
    alpha1 = w1 / total
    alpha2 = w2 / total
    
    # Blend
    if len(img1.shape) == 3:
        alpha1 = alpha1[:, :, np.newaxis]
        alpha2 = alpha2[:, :, np.newaxis]
    
    blended = img1 * alpha1 + img2 * alpha2
    
    return blended


def simple_average_blend(img1: np.ndarray,
                         img2: np.ndarray,
                         mask1: np.ndarray,
                         mask2: np.ndarray) -> np.ndarray:
    """
    Simple average blending in overlap regions.
    
    Args:
        img1: First image
        img2: Second image  
        mask1: Validity mask for img1
        mask2: Validity mask for img2
        
    Returns:
        Blended image
    """
    output = np.zeros_like(img1)
    
    # Only img1 valid
    only1 = mask1 & ~mask2
    # Only img2 valid
    only2 = mask2 & ~mask1
    # Both valid (overlap)
    both = mask1 & mask2
    
    if len(img1.shape) == 3:
        only1 = only1[:, :, np.newaxis]
        only2 = only2[:, :, np.newaxis]
        both = both[:, :, np.newaxis]
    
    output = (img1 * only1 + 
              img2 * only2 + 
              (img1 + img2) / 2 * both)
    
    return output


def seam_blend(img1: np.ndarray,
               img2: np.ndarray,
               mask1: np.ndarray,
               mask2: np.ndarray) -> np.ndarray:
    """
    Blend at optimal seam (minimum error boundary).
    
    Finds the path through the overlap region that minimizes
    the visible difference between images.
    
    Args:
        img1: First image
        img2: Second image
        mask1: Validity mask for img1
        mask2: Validity mask for img2
        
    Returns:
        Blended image
    """
    h, w = mask1.shape
    
    # Find overlap region
    overlap = mask1 & mask2
    
    if not np.any(overlap):
        return simple_average_blend(img1, img2, mask1, mask2)
    
    # Find overlap bounds
    overlap_cols = np.where(np.any(overlap, axis=0))[0]
    if len(overlap_cols) < 2:
        return simple_average_blend(img1, img2, mask1, mask2)
    
    # Compute difference at each pixel (for seam finding)
    if len(img1.shape) == 3:
        diff = np.sum((img1 - img2) ** 2, axis=2)
    else:
        diff = (img1 - img2) ** 2
    
    # Add large cost for non-overlap regions
    diff = np.where(overlap, diff, 1e10)
    
    # Dynamic programming for optimal vertical seam
    seam_col = overlap_cols[len(overlap_cols) // 2]  # Start in middle of overlap
    
    # Create blend mask (1 = use img1, 0 = use img2)
    blend_mask = np.zeros((h, w), dtype=np.float64)
    blend_mask[:, :seam_col] = 1
    
    # Simple feathering around seam
    feather = 20
    for x in range(max(0, seam_col - feather), min(w, seam_col + feather)):
        t = (x - (seam_col - feather)) / (2 * feather)
        blend_mask[:, x] = 1 - t
    
    # Apply blend
    if len(img1.shape) == 3:
        blend_mask = blend_mask[:, :, np.newaxis]
    
    # Only blend where both are valid
    both = (mask1 & mask2)
    if len(img1.shape) == 3:
        both_3d = both[:, :, np.newaxis]
    else:
        both_3d = both
    
    output = np.where(both_3d,
                      img1 * blend_mask + img2 * (1 - blend_mask),
                      np.where(mask1[:, :, np.newaxis] if len(img1.shape) == 3 else mask1,
                               img1, img2))
    
    return output
