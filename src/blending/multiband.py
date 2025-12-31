"""
Multi-band blending using Laplacian pyramids.
High-quality blending that preserves details at all scales.
"""

import numpy as np
from typing import List, Tuple
from ..core.convolution import gaussian_blur


def reduce_image(image: np.ndarray) -> np.ndarray:
    """
    Reduce image size by half (blur + subsample).
    
    Args:
        image: Input image
        
    Returns:
        Reduced image (half size)
    """
    # Apply Gaussian blur
    blurred = gaussian_blur(image, sigma=1.0)
    
    # Subsample (take every other pixel)
    if len(image.shape) == 3:
        return blurred[::2, ::2, :]
    return blurred[::2, ::2]


def expand_image(image: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """
    Expand image to target size (upsample + blur).
    
    Args:
        image: Input image
        target_shape: (height, width) of output
        
    Returns:
        Expanded image
    """
    h, w = image.shape[:2]
    target_h, target_w = target_shape
    
    # Create upsampled image (zeros between pixels)
    if len(image.shape) == 3:
        expanded = np.zeros((target_h, target_w, image.shape[2]), dtype=np.float64)
        expanded[::2, ::2, :] = image[:min(h, (target_h+1)//2), :min(w, (target_w+1)//2), :]
    else:
        expanded = np.zeros((target_h, target_w), dtype=np.float64)
        expanded[::2, ::2] = image[:min(h, (target_h+1)//2), :min(w, (target_w+1)//2)]
    
    # Scale by 4 to compensate for zeros
    expanded *= 4
    
    # Apply Gaussian blur to interpolate
    expanded = gaussian_blur(expanded, sigma=1.0)
    
    return expanded


def gaussian_pyramid(image: np.ndarray, levels: int) -> List[np.ndarray]:
    """
    Build Gaussian pyramid (successive low-pass filtering + downsampling).
    
    Level 0 is the original image.
    Each subsequent level is half the size of the previous.
    
    Args:
        image: Input image
        levels: Number of pyramid levels
        
    Returns:
        List of images from finest to coarsest
    """
    pyramid = [image.astype(np.float64)]
    
    for _ in range(levels - 1):
        current = pyramid[-1]
        
        # Check if we can reduce further
        if current.shape[0] < 2 or current.shape[1] < 2:
            break
        
        reduced = reduce_image(current)
        pyramid.append(reduced)
    
    return pyramid


def laplacian_pyramid(image: np.ndarray, levels: int) -> List[np.ndarray]:
    """
    Build Laplacian pyramid (band-pass decomposition).
    
    Each level contains the difference between consecutive 
    Gaussian pyramid levels - essentially the details at that scale.
    
    The last level is a low-pass residual.
    
    Args:
        image: Input image
        levels: Number of pyramid levels
        
    Returns:
        List of Laplacian images (details at each scale)
    """
    # First build Gaussian pyramid
    gauss_pyr = gaussian_pyramid(image, levels)
    
    laplacian = []
    
    for i in range(len(gauss_pyr) - 1):
        # Expand the lower resolution level
        expanded = expand_image(gauss_pyr[i + 1], gauss_pyr[i].shape[:2])
        
        # Subtract to get the details
        diff = gauss_pyr[i] - expanded
        laplacian.append(diff)
    
    # Last level is the residual (lowest frequency)
    laplacian.append(gauss_pyr[-1])
    
    return laplacian


def reconstruct_from_laplacian(pyramid: List[np.ndarray]) -> np.ndarray:
    """
    Reconstruct image from its Laplacian pyramid.
    
    Start from the coarsest level and successively add details.
    
    Args:
        pyramid: Laplacian pyramid (finest to coarsest)
        
    Returns:
        Reconstructed image
    """
    # Start with the coarsest level
    image = pyramid[-1].copy()
    
    # Add details from coarse to fine
    for i in range(len(pyramid) - 2, -1, -1):
        # Expand current image
        expanded = expand_image(image, pyramid[i].shape[:2])
        
        # Add details
        image = expanded + pyramid[i]
    
    return image


def blend_pyramids(lap1: List[np.ndarray], 
                   lap2: List[np.ndarray],
                   mask_pyr: List[np.ndarray]) -> List[np.ndarray]:
    """
    Blend two Laplacian pyramids using a mask pyramid.
    
    At each level: blended = lap1 * mask + lap2 * (1 - mask)
    
    Args:
        lap1: Laplacian pyramid of image 1
        lap2: Laplacian pyramid of image 2
        mask_pyr: Gaussian pyramid of the blend mask
        
    Returns:
        Blended Laplacian pyramid
    """
    blended = []
    
    for l1, l2, m in zip(lap1, lap2, mask_pyr):
        # Ensure mask has right dimensions
        if len(l1.shape) == 3 and len(m.shape) == 2:
            m = m[:, :, np.newaxis]
        
        # Blend at this level
        b = l1 * m + l2 * (1 - m)
        blended.append(b)
    
    return blended


def multiband_blend(img1: np.ndarray,
                    img2: np.ndarray,
                    mask: np.ndarray,
                    levels: int = 5) -> np.ndarray:
    """
    Multi-band blending using Laplacian pyramids.
    
    This technique (Burt & Adelson, 1983) blends images at multiple
    frequency bands, smoothly transitioning low frequencies while
    preserving high-frequency details.
    
    This produces much smoother blends than simple alpha blending,
    especially for images with different exposures.
    
    Args:
        img1: First image
        img2: Second image (same size as img1)
        mask: Blend mask (1 = img1, 0 = img2)
        levels: Number of pyramid levels
        
    Returns:
        Blended image
    """
    # Ensure images are float
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mask = mask.astype(np.float64)
    
    # Ensure same shape
    assert img1.shape == img2.shape, "Images must have same shape"
    
    # Clamp levels based on image size
    min_dim = min(img1.shape[0], img1.shape[1])
    max_levels = int(np.log2(min_dim)) - 1
    levels = min(levels, max_levels)
    
    if levels < 2:
        # Fall back to simple blend
        if len(img1.shape) == 3:
            mask = mask[:, :, np.newaxis]
        return img1 * mask + img2 * (1 - mask)
    
    # Build Laplacian pyramids for both images
    lap1 = laplacian_pyramid(img1, levels)
    lap2 = laplacian_pyramid(img2, levels)
    
    # Build Gaussian pyramid for the mask
    mask_pyr = gaussian_pyramid(mask, levels)
    
    # Blend pyramids
    lap_blended = blend_pyramids(lap1, lap2, mask_pyr)
    
    # Reconstruct
    result = reconstruct_from_laplacian(lap_blended)
    
    # Clip to valid range
    result = np.clip(result, 0, 1)
    
    return result


def multiband_blend_with_masks(img1: np.ndarray,
                                img2: np.ndarray,
                                mask1: np.ndarray,
                                mask2: np.ndarray,
                                levels: int = 5) -> np.ndarray:
    """
    Multi-band blend using validity masks.
    
    Creates appropriate blend mask from two validity masks with
    smooth distance-based transitions.
    
    Args:
        img1: First image
        img2: Second image
        mask1: Validity mask for img1
        mask2: Validity mask for img2
        levels: Pyramid levels
        
    Returns:
        Blended image
    """
    h, w = img1.shape[:2]
    
    # Find overlap region
    overlap = mask1 & mask2
    only1 = mask1 & ~mask2
    only2 = mask2 & ~mask1
    
    # Create smooth blend mask using distance transform
    blend_mask = np.zeros((h, w), dtype=np.float64)
    
    # Set non-overlapping regions
    blend_mask[only1] = 1.0
    blend_mask[only2] = 0.0
    
    if np.any(overlap):
        # Compute distance from each valid region's edge
        # For smooth blending, we use distance-based weights
        
        # Distance from img1-only region
        dist1 = _compute_distance_field(mask1 & ~mask2, overlap)
        
        # Distance from img2-only region  
        dist2 = _compute_distance_field(mask2 & ~mask1, overlap)
        
        # Blend weight is based on relative distances
        # Points closer to img1 region get higher weight for img1
        total_dist = dist1 + dist2
        total_dist = np.where(total_dist < 1e-10, 1.0, total_dist)
        
        # Weight for img1 in overlap region
        weight1 = dist2 / total_dist  # Further from img2 = more img1
        
        blend_mask[overlap] = weight1[overlap]
        
        # Apply Gaussian smoothing to the mask for even smoother transitions
        blend_mask = gaussian_blur(blend_mask, sigma=3.0)
        
        # Restore definite regions after blur
        blend_mask[only1] = 1.0
        blend_mask[only2] = 0.0
    
    # Apply multi-band blend
    return multiband_blend(img1, img2, blend_mask, levels)


def _compute_distance_field(source_mask: np.ndarray, 
                             target_mask: np.ndarray) -> np.ndarray:
    """
    Compute distance from source region to all points in target region.
    
    Uses iterative erosion as approximation of distance transform.
    
    Args:
        source_mask: Binary mask of source region
        target_mask: Binary mask where we compute distances
        
    Returns:
        Distance field (higher = further from source)
    """
    h, w = source_mask.shape
    
    # Initialize distance field
    dist = np.zeros((h, w), dtype=np.float64)
    
    # Start from source boundary
    current = source_mask.astype(np.float64)
    
    # Grow outward and record distances
    max_iter = min(200, min(h, w) // 2)
    
    for iteration in range(1, max_iter + 1):
        # Dilate current region
        dilated = np.zeros_like(current)
        dilated[1:-1, 1:-1] = np.maximum.reduce([
            current[1:-1, 1:-1],
            current[:-2, 1:-1],   # top
            current[2:, 1:-1],    # bottom
            current[1:-1, :-2],   # left
            current[1:-1, 2:]     # right
        ])
        
        # Find newly reached pixels in target
        newly_reached = (dilated > 0) & (current == 0) & target_mask
        
        # Set their distance
        dist[newly_reached] = iteration
        
        # Update current
        current = dilated
        
        # Stop if we've covered all of target
        if np.all(dist[target_mask] > 0):
            break
    
    # Handle any remaining unreached pixels
    unreached = target_mask & (dist == 0)
    if np.any(unreached):
        dist[unreached] = max_iter + 1
    
    return dist


def exposure_compensate(img1: np.ndarray,
                        img2: np.ndarray,
                        mask1: np.ndarray,
                        mask2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compensate for exposure differences between images.
    
    Adjusts the brightness/contrast of one image to match the overlap
    region of the other.
    
    Args:
        img1: First image
        img2: Second image
        mask1: Validity mask for img1
        mask2: Validity mask for img2
        
    Returns:
        (adjusted_img1, adjusted_img2)
    """
    # Find overlap
    overlap = mask1 & mask2
    
    if not np.any(overlap):
        return img1, img2
    
    # Compute mean intensity in overlap for each image
    if len(img1.shape) == 3:
        overlap_3d = overlap[:, :, np.newaxis]
        mean1 = np.sum(img1 * overlap_3d) / (3 * np.sum(overlap) + 1e-10)
        mean2 = np.sum(img2 * overlap_3d) / (3 * np.sum(overlap) + 1e-10)
    else:
        mean1 = np.sum(img1 * overlap) / (np.sum(overlap) + 1e-10)
        mean2 = np.sum(img2 * overlap) / (np.sum(overlap) + 1e-10)
    
    # Compute adjustment factor
    target_mean = (mean1 + mean2) / 2
    
    if mean1 > 1e-10:
        scale1 = target_mean / mean1
    else:
        scale1 = 1.0
    
    if mean2 > 1e-10:
        scale2 = target_mean / mean2
    else:
        scale2 = 1.0
    
    # Limit adjustment range
    scale1 = np.clip(scale1, 0.5, 2.0)
    scale2 = np.clip(scale2, 0.5, 2.0)
    
    # Apply adjustment
    adjusted1 = np.clip(img1 * scale1, 0, 1)
    adjusted2 = np.clip(img2 * scale2, 0, 1)
    
    return adjusted1, adjusted2
