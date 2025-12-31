"""
Feature descriptor implementation.
Computes binary descriptors similar to BRIEF/ORB for efficient matching.
"""

import numpy as np
from typing import List, Tuple, Optional
from .harris import Keypoint
from ..core.convolution import gaussian_blur, sobel_gradients


# Pre-defined sampling pattern for BRIEF-like descriptor
# These are relative coordinates (dx, dy) for pixel pairs to compare
def generate_brief_pattern(n_pairs: int = 256, patch_size: int = 31, 
                           seed: int = 42) -> np.ndarray:
    """
    Generate a sampling pattern for BRIEF descriptor.
    
    The pattern defines pairs of points within a patch. For each pair,
    we compare the intensities to get a binary bit.
    
    Args:
        n_pairs: Number of pairs (each pair gives 1 bit)
        patch_size: Size of the patch around keypoint
        seed: Random seed for reproducibility
        
    Returns:
        Array of shape (n_pairs, 4) with (x1, y1, x2, y2) for each pair
    """
    np.random.seed(seed)
    half = patch_size // 2
    
    # Sample from isotropic Gaussian with sigma = patch_size/5
    sigma = patch_size / 5
    
    # Generate random points using Gaussian distribution
    pattern = np.zeros((n_pairs, 4), dtype=np.int32)
    
    for i in range(n_pairs):
        # First point
        x1 = int(np.clip(np.random.randn() * sigma, -half, half))
        y1 = int(np.clip(np.random.randn() * sigma, -half, half))
        
        # Second point
        x2 = int(np.clip(np.random.randn() * sigma, -half, half))
        y2 = int(np.clip(np.random.randn() * sigma, -half, half))
        
        pattern[i] = [x1, y1, x2, y2]
    
    return pattern


# Global pattern (computed once)
_BRIEF_PATTERN = None


def get_brief_pattern() -> np.ndarray:
    """Get the global BRIEF pattern (lazy initialization)."""
    global _BRIEF_PATTERN
    if _BRIEF_PATTERN is None:
        _BRIEF_PATTERN = generate_brief_pattern()
    return _BRIEF_PATTERN


def compute_orientation(image: np.ndarray, kp: Keypoint, 
                        radius: int = 15) -> float:
    """
    Compute dominant orientation of a keypoint using intensity centroid.
    
    The intensity centroid is the center of mass of the patch weighted
    by pixel intensities. The orientation is the angle from the keypoint
    to the centroid.
    
    This makes the descriptor rotation-invariant.
    
    Args:
        image: Grayscale image
        kp: Keypoint to compute orientation for
        radius: Radius of circular patch
        
    Returns:
        Orientation in radians [-pi, pi]
    """
    h, w = image.shape[:2]
    x, y = int(kp.x), int(kp.y)
    
    # Bounds check
    if x - radius < 0 or x + radius >= w or y - radius < 0 or y + radius >= h:
        return 0.0
    
    # Extract patch
    patch = image[y - radius:y + radius + 1, x - radius:x + radius + 1]
    
    # Create coordinate grids centered at (radius, radius)
    yy, xx = np.ogrid[-radius:radius+1, -radius:radius+1]
    
    # Circular mask
    mask = (xx ** 2 + yy ** 2) <= radius ** 2
    
    # Compute intensity centroid
    m10 = np.sum(xx * patch * mask)  # Sum of x * intensity
    m01 = np.sum(yy * patch * mask)  # Sum of y * intensity
    
    # Angle from center to centroid
    return np.arctan2(m01, m10)


def rotate_pattern(pattern: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate sampling pattern by given angle.
    
    Args:
        pattern: (n_pairs, 4) array of sampling coordinates
        angle: Rotation angle in radians
        
    Returns:
        Rotated pattern
    """
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    
    rotated = np.zeros_like(pattern)
    
    # Rotate each point
    for i in range(pattern.shape[0]):
        x1, y1, x2, y2 = pattern[i]
        
        # Rotate first point
        rx1 = int(np.round(x1 * cos_a - y1 * sin_a))
        ry1 = int(np.round(x1 * sin_a + y1 * cos_a))
        
        # Rotate second point
        rx2 = int(np.round(x2 * cos_a - y2 * sin_a))
        ry2 = int(np.round(x2 * sin_a + y2 * cos_a))
        
        rotated[i] = [rx1, ry1, rx2, ry2]
    
    return rotated


def extract_descriptor(image: np.ndarray, kp: Keypoint,
                       pattern: np.ndarray,
                       use_orientation: bool = True) -> Optional[np.ndarray]:
    """
    Extract binary descriptor for a single keypoint.
    
    For each pair defined in the pattern, compare the intensities.
    If I(p1) < I(p2), bit = 1, else bit = 0.
    
    Args:
        image: Grayscale image (should be pre-smoothed)
        kp: Keypoint to describe
        pattern: Sampling pattern (n_pairs, 4)
        use_orientation: Whether to use orientation for rotation invariance
        
    Returns:
        Binary descriptor as uint8 array (32 bytes = 256 bits), or None if failed
    """
    h, w = image.shape[:2]
    x, y = int(kp.x), int(kp.y)
    
    # Get rotated pattern if using orientation
    if use_orientation:
        angle = kp.orientation
        pat = rotate_pattern(pattern, angle)
    else:
        pat = pattern
    
    # Check bounds (need margin for all pattern points)
    max_offset = np.abs(pattern).max() + 1
    if x - max_offset < 0 or x + max_offset >= w:
        return None
    if y - max_offset < 0 or y + max_offset >= h:
        return None
    
    # Compute binary descriptor
    n_pairs = pattern.shape[0]
    n_bytes = (n_pairs + 7) // 8  # Round up to bytes
    descriptor = np.zeros(n_bytes, dtype=np.uint8)
    
    for i in range(n_pairs):
        dx1, dy1, dx2, dy2 = pat[i]
        
        # Sample intensities
        try:
            i1 = image[y + dy1, x + dx1]
            i2 = image[y + dy2, x + dx2]
        except IndexError:
            return None
        
        # Set bit if i1 < i2
        if i1 < i2:
            byte_idx = i // 8
            bit_idx = i % 8
            descriptor[byte_idx] |= (1 << bit_idx)
    
    return descriptor


def compute_descriptors(image: np.ndarray, 
                        keypoints: List[Keypoint],
                        use_orientation: bool = True) -> Tuple[List[Keypoint], np.ndarray]:
    """
    Compute descriptors for all keypoints.
    
    Args:
        image: Input image (will be converted to grayscale if needed)
        keypoints: List of keypoints
        use_orientation: Whether to make descriptors rotation-invariant
        
    Returns:
        (valid_keypoints, descriptors)
        - valid_keypoints: Keypoints for which descriptors were computed
        - descriptors: (N, 32) array of uint8 descriptors
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    else:
        gray = image.copy()
    
    # Apply Gaussian blur to reduce noise
    gray = gaussian_blur(gray, sigma=2.0)
    
    # Get sampling pattern
    pattern = get_brief_pattern()
    
    # Compute orientation for each keypoint
    if use_orientation:
        for kp in keypoints:
            kp.orientation = compute_orientation(gray, kp)
    
    # Extract descriptors
    valid_kps = []
    descriptors = []
    
    for kp in keypoints:
        desc = extract_descriptor(gray, kp, pattern, use_orientation)
        
        if desc is not None:
            valid_kps.append(kp)
            descriptors.append(desc)
    
    if not descriptors:
        return [], np.array([]).reshape(0, 32).astype(np.uint8)
    
    return valid_kps, np.array(descriptors, dtype=np.uint8)


def visualize_keypoints(image: np.ndarray, 
                        keypoints: List[Keypoint],
                        show_orientation: bool = True) -> np.ndarray:
    """
    Draw keypoints on image for visualization.
    
    Args:
        image: Input image
        keypoints: List of keypoints to draw
        show_orientation: Whether to draw orientation lines
        
    Returns:
        Image with keypoints drawn
    """
    # Ensure RGB
    if len(image.shape) == 2:
        vis = np.stack([image] * 3, axis=-1)
    else:
        vis = image.copy()
    
    # Make sure it's in [0, 1]
    if vis.max() > 1:
        vis = vis / 255.0
    
    # Draw each keypoint
    for kp in keypoints:
        x, y = int(kp.x), int(kp.y)
        
        # Draw circle
        radius = max(3, int(kp.scale * 3))
        for angle in np.linspace(0, 2 * np.pi, 20):
            cx = int(x + radius * np.cos(angle))
            cy = int(y + radius * np.sin(angle))
            if 0 <= cx < vis.shape[1] and 0 <= cy < vis.shape[0]:
                vis[cy, cx] = [0, 1, 0]  # Green
        
        # Draw center
        if 0 <= x < vis.shape[1] and 0 <= y < vis.shape[0]:
            vis[y, x] = [0, 1, 0]
        
        # Draw orientation line
        if show_orientation and kp.orientation != 0:
            end_x = int(x + radius * np.cos(kp.orientation))
            end_y = int(y + radius * np.sin(kp.orientation))
            
            # Simple line drawing
            steps = max(abs(end_x - x), abs(end_y - y), 1)
            for t in range(steps + 1):
                lx = int(x + t * (end_x - x) / steps)
                ly = int(y + t * (end_y - y) / steps)
                if 0 <= lx < vis.shape[1] and 0 <= ly < vis.shape[0]:
                    vis[ly, lx] = [1, 0, 0]  # Red
    
    return vis
