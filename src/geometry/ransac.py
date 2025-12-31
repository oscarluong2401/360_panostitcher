"""
RANSAC (Random Sample Consensus) for robust estimation.
Handles outliers in feature matches.
"""

import numpy as np
from typing import Tuple, Optional, Callable
from .homography import compute_homography_dlt, compute_reprojection_error, refine_homography


def ransac_homography(src_points: np.ndarray,
                      dst_points: np.ndarray,
                      n_iterations: int = 1000,
                      threshold: float = 5.0,
                      min_inliers: int = 10,
                      confidence: float = 0.99) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """
    RANSAC for robust homography estimation.
    
    RANSAC (Random Sample Consensus) is designed to handle outliers.
    It works by:
    1. Randomly sample minimal set (4 points for homography)
    2. Fit model to sample
    3. Count inliers (points with small error)
    4. Repeat, keeping the model with most inliers
    5. Refit using all inliers
    
    The algorithm adapts the number of iterations based on the inlier ratio.
    
    Args:
        src_points: Nx2 source points
        dst_points: Nx2 destination points
        n_iterations: Maximum number of iterations
        threshold: Inlier threshold in pixels
        min_inliers: Minimum inliers required for valid model
        confidence: Confidence level for adaptive iteration count
        
    Returns:
        (H, inlier_mask)
        - H: Best homography matrix (or None if failed)
        - inlier_mask: Boolean array indicating inliers
    """
    n = len(src_points)
    
    if n < 4:
        return None, np.zeros(n, dtype=bool)
    
    best_H = None
    best_inlier_mask = np.zeros(n, dtype=bool)
    best_n_inliers = 0
    
    # Adaptive iteration count
    adaptive_iterations = n_iterations
    
    for iteration in range(n_iterations):
        if iteration >= adaptive_iterations:
            break
        
        # Step 1: Randomly sample 4 points
        indices = np.random.choice(n, 4, replace=False)
        
        # Check if points are not collinear
        sample_src = src_points[indices]
        sample_dst = dst_points[indices]
        
        if _are_collinear(sample_src) or _are_collinear(sample_dst):
            continue
        
        # Step 2: Compute homography from sample
        try:
            H = compute_homography_dlt(sample_src, sample_dst)
        except:
            continue
        
        # Check if H is valid
        if not _is_valid_homography(H):
            continue
        
        # Step 3: Count inliers
        errors = compute_reprojection_error(H, src_points, dst_points)
        inlier_mask = errors < threshold
        n_inliers = np.sum(inlier_mask)
        
        # Step 4: Update best if this is better
        if n_inliers > best_n_inliers:
            best_n_inliers = n_inliers
            best_inlier_mask = inlier_mask.copy()
            best_H = H.copy()
            
            # Update adaptive iteration count
            inlier_ratio = n_inliers / n
            if inlier_ratio > 0:
                # Number of iterations needed to see all inliers with probability p
                # k = log(1-p) / log(1 - w^s)
                # where w = inlier ratio, s = sample size (4 for homography)
                w = inlier_ratio
                p = confidence
                k = np.log(1 - p) / np.log(1 - w**4 + 1e-10)
                adaptive_iterations = min(adaptive_iterations, int(k) + 1)
    
    # Check if we found enough inliers
    if best_n_inliers < min_inliers:
        return None, np.zeros(n, dtype=bool)
    
    # Step 5: Recompute homography using all inliers
    inlier_src = src_points[best_inlier_mask]
    inlier_dst = dst_points[best_inlier_mask]
    
    try:
        best_H = compute_homography_dlt(inlier_src, inlier_dst)
        
        # Optionally refine
        best_H = refine_homography(best_H, inlier_src, inlier_dst)
        
        # Update inlier mask with refined homography
        errors = compute_reprojection_error(best_H, src_points, dst_points)
        best_inlier_mask = errors < threshold
        
    except:
        pass
    
    return best_H, best_inlier_mask


def _are_collinear(points: np.ndarray, epsilon: float = 1e-6) -> bool:
    """
    Check if a set of points are collinear.
    
    Args:
        points: Nx2 array of points
        epsilon: Threshold for collinearity
        
    Returns:
        True if points are collinear
    """
    if len(points) < 3:
        return True
    
    # Use cross product to check collinearity
    # If all cross products are near zero, points are collinear
    p0 = points[0]
    
    for i in range(1, len(points) - 1):
        v1 = points[i] - p0
        v2 = points[i + 1] - p0
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        if abs(cross) > epsilon:
            return False
    
    return True


def _is_valid_homography(H: np.ndarray) -> bool:
    """
    Check if homography matrix is geometrically valid.
    
    Invalid cases include:
    - Near-singular matrix
    - Extreme perspective (flipping)
    
    Args:
        H: 3x3 homography matrix
        
    Returns:
        True if valid
    """
    # Check condition number (should not be too large)
    try:
        cond = np.linalg.cond(H)
        if cond > 1e8:
            return False
    except:
        return False
    
    # Check determinant (should be positive for orientation-preserving)
    det = np.linalg.det(H)
    if det < 1e-10:
        return False
    
    # Check for extreme scaling
    H_norm = H / H[2, 2] if abs(H[2, 2]) > 1e-10 else H
    
    # Upper-left 2x2 should have reasonable singular values
    upper_left = H_norm[:2, :2]
    try:
        s = np.linalg.svd(upper_left, compute_uv=False)
        if s[0] / s[1] > 100 or s[1] < 0.01 or s[0] > 100:
            return False
    except:
        return False
    
    return True


def prosac_homography(src_points: np.ndarray,
                      dst_points: np.ndarray,
                      match_scores: np.ndarray,
                      n_iterations: int = 1000,
                      threshold: float = 5.0) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """
    PROSAC (Progressive Sample Consensus) for homography estimation.
    
    PROSAC is like RANSAC but samples preferentially from high-quality
    matches first. This often converges faster than pure random sampling.
    
    Args:
        src_points: Nx2 source points
        dst_points: Nx2 destination points
        match_scores: Quality score for each match (higher = better)
        n_iterations: Maximum iterations
        threshold: Inlier threshold
        
    Returns:
        (H, inlier_mask)
    """
    n = len(src_points)
    
    if n < 4:
        return None, np.zeros(n, dtype=bool)
    
    # Sort by quality
    order = np.argsort(-match_scores)  # Descending order
    src_sorted = src_points[order]
    dst_sorted = dst_points[order]
    
    best_H = None
    best_inlier_mask = np.zeros(n, dtype=bool)
    best_n_inliers = 0
    
    # Progressive sampling: start with top matches
    sample_size = 4
    
    for i in range(n_iterations):
        # Expand sample pool progressively
        pool_size = min(n, sample_size + i // 10)
        
        # Sample from current pool
        if pool_size < 4:
            continue
            
        indices = np.random.choice(pool_size, 4, replace=False)
        
        sample_src = src_sorted[indices]
        sample_dst = dst_sorted[indices]
        
        if _are_collinear(sample_src) or _are_collinear(sample_dst):
            continue
        
        try:
            H = compute_homography_dlt(sample_src, sample_dst)
        except:
            continue
        
        if not _is_valid_homography(H):
            continue
        
        # Count inliers on ALL points (not just pool)
        errors = compute_reprojection_error(H, src_sorted, dst_sorted)
        inlier_mask = errors < threshold
        n_inliers = np.sum(inlier_mask)
        
        if n_inliers > best_n_inliers:
            best_n_inliers = n_inliers
            # Map back to original order
            best_inlier_mask = np.zeros(n, dtype=bool)
            best_inlier_mask[order] = inlier_mask
            best_H = H.copy()
    
    # Refine with all inliers
    if best_n_inliers >= 4:
        inlier_src = src_points[best_inlier_mask]
        inlier_dst = dst_points[best_inlier_mask]
        best_H = compute_homography_dlt(inlier_src, inlier_dst)
    
    return best_H, best_inlier_mask


def lmeds_homography(src_points: np.ndarray,
                     dst_points: np.ndarray,
                     n_iterations: int = 1000) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """
    Least Median of Squares (LMedS) for homography estimation.
    
    Instead of counting inliers, LMedS minimizes the median error.
    This is useful when the inlier ratio is > 50%.
    
    Args:
        src_points: Nx2 source points
        dst_points: Nx2 destination points
        n_iterations: Number of iterations
        
    Returns:
        (H, inlier_mask)
    """
    n = len(src_points)
    
    if n < 4:
        return None, np.zeros(n, dtype=bool)
    
    best_H = None
    best_median_error = float('inf')
    
    for _ in range(n_iterations):
        # Random sample
        indices = np.random.choice(n, 4, replace=False)
        
        sample_src = src_points[indices]
        sample_dst = dst_points[indices]
        
        if _are_collinear(sample_src):
            continue
        
        try:
            H = compute_homography_dlt(sample_src, sample_dst)
        except:
            continue
        
        if not _is_valid_homography(H):
            continue
        
        # Compute median error
        errors = compute_reprojection_error(H, src_points, dst_points)
        median_error = np.median(errors)
        
        if median_error < best_median_error:
            best_median_error = median_error
            best_H = H.copy()
    
    if best_H is None:
        return None, np.zeros(n, dtype=bool)
    
    # Determine inliers based on median absolute deviation
    errors = compute_reprojection_error(best_H, src_points, dst_points)
    sigma = 1.4826 * np.median(errors)  # Robust scale estimate
    threshold = 2.5 * sigma
    
    inlier_mask = errors < threshold
    
    # Refine with inliers
    if np.sum(inlier_mask) >= 4:
        best_H = compute_homography_dlt(
            src_points[inlier_mask], 
            dst_points[inlier_mask]
        )
    
    return best_H, inlier_mask
