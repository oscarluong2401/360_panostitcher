"""
Non-Maximum Suppression for feature detection.
Keeps only local maxima to avoid clustered detections.
"""

import numpy as np
from typing import List
from .harris import Keypoint


def non_maximum_suppression(keypoints: List[Keypoint],
                            radius: int = 10,
                            max_keypoints: int = 500) -> List[Keypoint]:
    """
    Non-maximum suppression to remove clustered keypoints.
    
    For each keypoint, suppress all weaker keypoints within a given radius.
    This ensures detected features are well-distributed across the image.
    
    Args:
        keypoints: List of detected keypoints
        radius: Suppression radius in pixels
        max_keypoints: Maximum number of keypoints to return
        
    Returns:
        Filtered list of keypoints
    """
    if not keypoints:
        return []
    
    # Sort by response (strongest first)
    sorted_kps = sorted(keypoints, key=lambda kp: kp.response, reverse=True)
    
    # Keep track of which keypoints are suppressed
    suppressed = [False] * len(sorted_kps)
    result = []
    
    radius_sq = radius ** 2
    
    for i, kp in enumerate(sorted_kps):
        if suppressed[i]:
            continue
        
        # Keep this keypoint
        result.append(kp)
        
        if len(result) >= max_keypoints:
            break
        
        # Suppress all weaker keypoints within radius
        for j in range(i + 1, len(sorted_kps)):
            if suppressed[j]:
                continue
            
            # Check distance
            dx = sorted_kps[j].x - kp.x
            dy = sorted_kps[j].y - kp.y
            dist_sq = dx * dx + dy * dy
            
            if dist_sq < radius_sq:
                suppressed[j] = True
    
    return result


def adaptive_non_maximum_suppression(keypoints: List[Keypoint],
                                      n_keypoints: int = 500,
                                      c_robust: float = 0.9) -> List[Keypoint]:
    """
    Adaptive Non-Maximum Suppression (ANMS).
    
    Selects keypoints that are well-distributed across the image
    by computing the minimum suppression radius for each keypoint.
    
    The suppression radius r_i for keypoint i is the distance to the
    nearest keypoint j with response > c_robust * response_i.
    
    This tends to give better spatial distribution than simple NMS.
    
    Args:
        keypoints: List of detected keypoints
        n_keypoints: Desired number of keypoints
        c_robust: Robustness factor (typically 0.9)
        
    Returns:
        Filtered list of keypoints
    """
    if len(keypoints) <= n_keypoints:
        return keypoints
    
    n = len(keypoints)
    
    # Compute suppression radius for each keypoint
    radii = np.full(n, np.inf)
    
    for i in range(n):
        kp_i = keypoints[i]
        threshold = c_robust * kp_i.response
        
        for j in range(n):
            if i == j:
                continue
            
            kp_j = keypoints[j]
            
            # Only consider keypoints with significantly higher response
            if kp_j.response > threshold:
                dx = kp_j.x - kp_i.x
                dy = kp_j.y - kp_i.y
                dist = np.sqrt(dx * dx + dy * dy)
                radii[i] = min(radii[i], dist)
    
    # Sort by suppression radius (largest first)
    indices = np.argsort(-radii)
    
    # Take top n_keypoints
    return [keypoints[i] for i in indices[:n_keypoints]]


def grid_based_selection(keypoints: List[Keypoint],
                         image_shape: tuple,
                         grid_size: int = 30,
                         max_per_cell: int = 5) -> List[Keypoint]:
    """
    Grid-based keypoint selection for uniform distribution.
    
    Divides the image into a grid and selects the top keypoints
    from each cell, ensuring good spatial coverage.
    
    Args:
        keypoints: List of detected keypoints
        image_shape: (height, width) of the image
        grid_size: Size of each grid cell in pixels
        max_per_cell: Maximum keypoints per cell
        
    Returns:
        Selected keypoints
    """
    if not keypoints:
        return []
    
    h, w = image_shape[:2]
    n_cells_y = max(1, h // grid_size)
    n_cells_x = max(1, w // grid_size)
    
    # Create grid of keypoint lists
    grid = [[[] for _ in range(n_cells_x)] for _ in range(n_cells_y)]
    
    # Assign keypoints to cells
    for kp in keypoints:
        cell_x = min(int(kp.x // grid_size), n_cells_x - 1)
        cell_y = min(int(kp.y // grid_size), n_cells_y - 1)
        grid[cell_y][cell_x].append(kp)
    
    # Select top keypoints from each cell
    result = []
    for row in grid:
        for cell in row:
            # Sort by response
            cell_sorted = sorted(cell, key=lambda kp: kp.response, reverse=True)
            result.extend(cell_sorted[:max_per_cell])
    
    return result


def subpixel_refinement(image: np.ndarray, 
                        keypoints: List[Keypoint],
                        window_size: int = 5) -> List[Keypoint]:
    """
    Refine keypoint locations to subpixel accuracy.
    
    Uses quadratic interpolation around the detected corner
    to find a more accurate position.
    
    Args:
        image: The response image or original image
        keypoints: Keypoints with integer coordinates
        window_size: Size of window for refinement
        
    Returns:
        Keypoints with refined coordinates
    """
    h, w = image.shape[:2]
    half = window_size // 2
    refined = []
    
    for kp in keypoints:
        x, y = int(kp.x), int(kp.y)
        
        # Skip if too close to edge
        if x < half or x >= w - half or y < half or y >= h - half:
            refined.append(kp)
            continue
        
        # Extract local patch
        patch = image[y-1:y+2, x-1:x+2]
        
        if patch.shape != (3, 3):
            refined.append(kp)
            continue
        
        # Fit 2D quadratic and find extremum
        # f(dx, dy) = a*dx^2 + b*dy^2 + c*dx*dy + d*dx + e*dy + f
        
        # Using finite differences
        dx = (patch[1, 2] - patch[1, 0]) / 2
        dy = (patch[2, 1] - patch[0, 1]) / 2
        dxx = patch[1, 2] + patch[1, 0] - 2 * patch[1, 1]
        dyy = patch[2, 1] + patch[0, 1] - 2 * patch[1, 1]
        dxy = (patch[2, 2] - patch[2, 0] - patch[0, 2] + patch[0, 0]) / 4
        
        # Solve for extremum: [dxx dxy] [dx']   [-dx]
        #                     [dxy dyy] [dy'] = [-dy]
        det = dxx * dyy - dxy * dxy
        
        if abs(det) < 1e-10:
            refined.append(kp)
            continue
        
        offset_x = -(dyy * dx - dxy * dy) / det
        offset_y = -(dxx * dy - dxy * dx) / det
        
        # Only add if offset is reasonable
        if abs(offset_x) < 1.0 and abs(offset_y) < 1.0:
            refined.append(Keypoint(
                x=kp.x + offset_x,
                y=kp.y + offset_y,
                response=kp.response,
                scale=kp.scale,
                orientation=kp.orientation
            ))
        else:
            refined.append(kp)
    
    return refined
