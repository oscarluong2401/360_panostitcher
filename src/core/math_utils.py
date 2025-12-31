"""
Math utility functions for panorama stitching.
All implementations from scratch without scipy/cv2.
"""

import numpy as np
from typing import Tuple, Optional


def normalize(arr: np.ndarray, new_min: float = 0.0, new_max: float = 1.0) -> np.ndarray:
    """
    Normalize array values to a new range.
    
    Args:
        arr: Input array
        new_min: Minimum value of output range
        new_max: Maximum value of output range
        
    Returns:
        Normalized array
    """
    arr_min = arr.min()
    arr_max = arr.max()
    
    if arr_max - arr_min == 0:
        return np.full_like(arr, (new_min + new_max) / 2, dtype=np.float64)
    
    normalized = (arr - arr_min) / (arr_max - arr_min)
    return normalized * (new_max - new_min) + new_min


def clamp(value: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    """
    Clamp values to a specified range.
    
    Args:
        value: Input array or scalar
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        
    Returns:
        Clamped array
    """
    return np.clip(value, min_val, max_val)


def to_homogeneous(points: np.ndarray) -> np.ndarray:
    """
    Convert 2D points to homogeneous coordinates.
    
    Args:
        points: Nx2 array of 2D points
        
    Returns:
        Nx3 array of homogeneous points
    """
    n = points.shape[0]
    return np.hstack([points, np.ones((n, 1))])


def from_homogeneous(points: np.ndarray) -> np.ndarray:
    """
    Convert homogeneous coordinates to 2D points.
    
    Args:
        points: Nx3 array of homogeneous points
        
    Returns:
        Nx2 array of 2D points
    """
    # Divide by last coordinate (perspective division)
    w = points[:, 2:3]
    # Avoid division by zero
    w = np.where(np.abs(w) < 1e-10, 1e-10, w)
    return points[:, :2] / w


def solve_linear_least_squares(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve linear least squares problem Ax = b using normal equations.
    
    Args:
        A: mxn matrix
        b: mx1 vector
        
    Returns:
        nx1 solution vector
    """
    # Normal equations: A^T A x = A^T b
    ATA = A.T @ A
    ATb = A.T @ b
    
    # Solve using numpy (still valid - not cv2!)
    return np.linalg.solve(ATA, ATb)


def svd_solve_null(A: np.ndarray) -> np.ndarray:
    """
    Find the null space of matrix A using SVD.
    Solution is the right singular vector corresponding to smallest singular value.
    
    This is used for solving homogeneous equations Ax = 0.
    
    Args:
        A: mxn matrix
        
    Returns:
        nx1 null space vector (unit norm)
    """
    # Using numpy's SVD - this is fundamental linear algebra, not CV
    U, S, Vt = np.linalg.svd(A)
    # The null space is the last row of Vt (corresponding to smallest singular value)
    return Vt[-1]


def condition_points(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Condition/normalize points for numerical stability in homography estimation.
    
    Applies similarity transform so that:
    - Centroid is at origin
    - Average distance from origin is sqrt(2)
    
    Args:
        points: Nx2 array of points
        
    Returns:
        (conditioned_points, conditioning_matrix)
    """
    # Compute centroid
    centroid = np.mean(points, axis=0)
    
    # Translate to origin
    centered = points - centroid
    
    # Compute average distance from origin
    distances = np.sqrt(np.sum(centered**2, axis=1))
    avg_dist = np.mean(distances)
    
    if avg_dist < 1e-10:
        avg_dist = 1e-10
    
    # Scale factor to make average distance sqrt(2)
    scale = np.sqrt(2) / avg_dist
    
    # Conditioning matrix (similarity transform)
    T = np.array([
        [scale, 0, -scale * centroid[0]],
        [0, scale, -scale * centroid[1]],
        [0, 0, 1]
    ])
    
    # Apply transformation
    conditioned = centered * scale
    
    return conditioned, T


def rotation_matrix_2d(angle: float) -> np.ndarray:
    """
    Create a 2D rotation matrix.
    
    Args:
        angle: Rotation angle in radians
        
    Returns:
        2x2 rotation matrix
    """
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


def euclidean_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """
    Compute Euclidean distance between two points.
    
    Args:
        p1: First point
        p2: Second point
        
    Returns:
        Euclidean distance
    """
    return np.sqrt(np.sum((p1 - p2) ** 2))


def distance_matrix(points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Euclidean distances between two sets of points.
    
    Args:
        points1: Nx2 array
        points2: Mx2 array
        
    Returns:
        NxM distance matrix
    """
    # Using broadcasting for efficiency
    # (N, 1, 2) - (1, M, 2) -> (N, M, 2)
    diff = points1[:, np.newaxis, :] - points2[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=2))
