"""
Homography estimation using Direct Linear Transform (DLT).
Computes perspective transformation between two images.
"""

import numpy as np
from typing import Tuple, Optional


def normalize_points(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize points for numerical stability in homography estimation.
    
    The normalization transforms points so that:
    - Centroid is at the origin
    - Average distance from origin is sqrt(2)
    
    This conditioning is critical for accurate DLT results.
    
    Args:
        points: Nx2 array of 2D points
        
    Returns:
        (normalized_points, normalization_matrix)
    """
    # Compute centroid
    centroid = np.mean(points, axis=0)
    
    # Center points at origin
    centered = points - centroid
    
    # Compute average distance from origin
    distances = np.sqrt(np.sum(centered ** 2, axis=1))
    avg_dist = np.mean(distances)
    
    if avg_dist < 1e-10:
        avg_dist = 1e-10
    
    # Scale factor
    scale = np.sqrt(2) / avg_dist
    
    # Normalization matrix (similarity transform)
    T = np.array([
        [scale, 0, -scale * centroid[0]],
        [0, scale, -scale * centroid[1]],
        [0, 0, 1]
    ], dtype=np.float64)
    
    normalized = centered * scale
    
    return normalized, T


def compute_homography_dlt(src_points: np.ndarray, 
                            dst_points: np.ndarray,
                            normalize: bool = True) -> np.ndarray:
    """
    Compute homography matrix using Direct Linear Transform (DLT).
    
    Given point correspondences (x, y) <-> (x', y'), we want to find
    the 3x3 homography matrix H such that:
    
        [x']     [h11 h12 h13] [x]
        [y'] = s [h21 h22 h23] [y]
        [1 ]     [h31 h32 h33] [1]
    
    where s is a scale factor (perspective division).
    
    The DLT algorithm:
    1. For each correspondence, create two equations
    2. Stack into matrix A
    3. Solve Ah = 0 using SVD (h is null space of A)
    4. Reshape h into 3x3 matrix
    
    Args:
        src_points: Nx2 source points
        dst_points: Nx2 destination points
        normalize: Whether to normalize points (highly recommended)
        
    Returns:
        3x3 homography matrix
        
    Raises:
        ValueError: If not enough points (need at least 4)
    """
    n = len(src_points)
    
    if n < 4:
        raise ValueError(f"Need at least 4 points, got {n}")
    
    # Normalize points for numerical stability
    if normalize:
        src_normalized, T_src = normalize_points(src_points)
        dst_normalized, T_dst = normalize_points(dst_points)
    else:
        src_normalized = src_points.copy()
        dst_normalized = dst_points.copy()
        T_src = np.eye(3)
        T_dst = np.eye(3)
    
    # Build matrix A (2n x 9)
    # For each point correspondence (x, y) -> (x', y'):
    # Row 1: [-x, -y, -1,  0,  0,  0,  x*x', y*x', x']
    # Row 2: [ 0,  0,  0, -x, -y, -1,  x*y', y*y', y']
    
    A = np.zeros((2 * n, 9), dtype=np.float64)
    
    for i in range(n):
        x, y = src_normalized[i]
        xp, yp = dst_normalized[i]
        
        A[2*i] = [-x, -y, -1, 0, 0, 0, x*xp, y*xp, xp]
        A[2*i + 1] = [0, 0, 0, -x, -y, -1, x*yp, y*yp, yp]
    
    # Solve using SVD
    # The solution h is the right singular vector corresponding to 
    # the smallest singular value (last column of V, or last row of Vt)
    try:
        U, S, Vt = np.linalg.svd(A)
        h = Vt[-1]
    except np.linalg.LinAlgError:
        return np.eye(3)
    
    # Reshape into 3x3 matrix
    H_normalized = h.reshape(3, 3)
    
    # Denormalize: H = T_dst^-1 @ H_normalized @ T_src
    if normalize:
        T_dst_inv = np.linalg.inv(T_dst)
        H = T_dst_inv @ H_normalized @ T_src
    else:
        H = H_normalized
    
    # Normalize so H[2,2] = 1 (if not zero)
    if abs(H[2, 2]) > 1e-10:
        H = H / H[2, 2]
    
    return H


def apply_homography(H: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Apply homography transformation to points.
    
    Args:
        H: 3x3 homography matrix
        points: Nx2 array of 2D points
        
    Returns:
        Nx2 array of transformed points
    """
    n = len(points)
    
    # Convert to homogeneous coordinates
    ones = np.ones((n, 1))
    points_h = np.hstack([points, ones])  # Nx3
    
    # Apply transformation
    transformed_h = (H @ points_h.T).T  # Nx3
    
    # Convert back from homogeneous (perspective division)
    w = transformed_h[:, 2:3]
    w = np.where(np.abs(w) < 1e-10, 1e-10, w)
    
    transformed = transformed_h[:, :2] / w
    
    return transformed


def compute_reprojection_error(H: np.ndarray, 
                                src_points: np.ndarray,
                                dst_points: np.ndarray) -> np.ndarray:
    """
    Compute reprojection error for each point correspondence.
    
    Error is the Euclidean distance between the mapped source point
    and the actual destination point.
    
    Args:
        H: 3x3 homography matrix
        src_points: Nx2 source points
        dst_points: Nx2 destination points
        
    Returns:
        N array of errors (one per point)
    """
    # Transform source points
    projected = apply_homography(H, src_points)
    
    # Compute Euclidean distance to destination
    errors = np.sqrt(np.sum((projected - dst_points) ** 2, axis=1))
    
    return errors


def symmetric_transfer_error(H: np.ndarray,
                              src_points: np.ndarray,
                              dst_points: np.ndarray) -> np.ndarray:
    """
    Compute symmetric transfer error.
    
    This measures error in both directions:
    - Forward: ||H(x) - x'||
    - Backward: ||H^-1(x') - x||
    
    The symmetric error is the sum of both.
    
    Args:
        H: 3x3 homography matrix
        src_points: Source points
        dst_points: Destination points
        
    Returns:
        N array of symmetric errors
    """
    # Forward error
    forward_error = compute_reprojection_error(H, src_points, dst_points)
    
    # Backward error
    try:
        H_inv = np.linalg.inv(H)
        backward_error = compute_reprojection_error(H_inv, dst_points, src_points)
    except np.linalg.LinAlgError:
        backward_error = forward_error
    
    return forward_error + backward_error


def refine_homography(H_initial: np.ndarray,
                      src_points: np.ndarray,
                      dst_points: np.ndarray,
                      max_iterations: int = 10) -> np.ndarray:
    """
    Refine homography using iterative reweighting.
    
    Points with high error are given lower weight in subsequent iterations.
    This is a simple robust refinement (not full Levenberg-Marquardt).
    
    Args:
        H_initial: Initial homography estimate
        src_points: Source points
        dst_points: Destination points
        max_iterations: Maximum refinement iterations
        
    Returns:
        Refined homography matrix
    """
    H = H_initial.copy()
    
    for _ in range(max_iterations):
        # Compute errors
        errors = compute_reprojection_error(H, src_points, dst_points)
        
        # Compute weights (inverse of error + small constant)
        sigma = np.median(errors) + 1e-10
        weights = 1.0 / (1.0 + (errors / sigma) ** 2)
        
        # Weighted DLT
        H = _weighted_dlt(src_points, dst_points, weights)
    
    return H


def _weighted_dlt(src_points: np.ndarray, 
                  dst_points: np.ndarray,
                  weights: np.ndarray) -> np.ndarray:
    """
    Weighted DLT for homography estimation.
    
    Args:
        src_points: Source points
        dst_points: Destination points
        weights: Per-point weights
        
    Returns:
        Homography matrix
    """
    n = len(src_points)
    
    # Normalize
    src_norm, T_src = normalize_points(src_points)
    dst_norm, T_dst = normalize_points(dst_points)
    
    # Build weighted matrix
    A = np.zeros((2 * n, 9), dtype=np.float64)
    
    for i in range(n):
        x, y = src_norm[i]
        xp, yp = dst_norm[i]
        w = np.sqrt(weights[i])  # Apply sqrt since we're minimizing squared error
        
        A[2*i] = w * np.array([-x, -y, -1, 0, 0, 0, x*xp, y*xp, xp])
        A[2*i + 1] = w * np.array([0, 0, 0, -x, -y, -1, x*yp, y*yp, yp])
    
    # Solve
    try:
        U, S, Vt = np.linalg.svd(A)
        h = Vt[-1]
    except np.linalg.LinAlgError:
        return np.eye(3)
    
    H_norm = h.reshape(3, 3)
    H = np.linalg.inv(T_dst) @ H_norm @ T_src
    
    if abs(H[2, 2]) > 1e-10:
        H = H / H[2, 2]
    
    return H


def decompose_homography(H: np.ndarray) -> dict:
    """
    Decompose homography into rotation, translation, and normal.
    
    This is an approximate decomposition assuming planar scene.
    Useful for understanding the geometric relationship.
    
    Args:
        H: 3x3 homography matrix
        
    Returns:
        Dictionary with 'rotation', 'translation', 'normal' (if valid)
    """
    # This is a simplified decomposition
    # Full decomposition requires knowing camera intrinsics
    
    # Compute SVD
    U, S, Vt = np.linalg.svd(H)
    
    # Scale factor from singular values
    scale = (S[0] * S[1] * S[2]) ** (1.0/3.0)
    
    # Approximate rotation
    R = U @ Vt
    
    # Ensure proper rotation (det = 1)
    if np.linalg.det(R) < 0:
        R = -R
    
    return {
        'rotation': R,
        'scale': scale,
        'singular_values': S
    }
