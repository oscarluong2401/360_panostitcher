"""
Feature matching algorithms.
Matches descriptors between images using various strategies.
"""

import numpy as np
from typing import List, Tuple, Optional
from .harris import Keypoint


class Match:
    """Represents a match between two keypoints."""
    
    def __init__(self, idx1: int, idx2: int, distance: int):
        self.queryIdx = idx1    # Index in first image
        self.trainIdx = idx2    # Index in second image
        self.distance = distance
    
    def __repr__(self) -> str:
        return f"Match({self.queryIdx} -> {self.trainIdx}, dist={self.distance})"


def hamming_distance(desc1: np.ndarray, desc2: np.ndarray) -> int:
    """
    Compute Hamming distance between two binary descriptors.
    
    Hamming distance = number of bits that differ.
    
    Args:
        desc1: First descriptor (uint8 array)
        desc2: Second descriptor (uint8 array)
        
    Returns:
        Hamming distance (number of different bits)
    """
    # XOR to find different bits
    xor_result = desc1 ^ desc2
    
    # Count set bits (popcount)
    # Using numpy's unpackbits to count
    bits = np.unpackbits(xor_result)
    return int(np.sum(bits))


def hamming_distance_fast(desc1: np.ndarray, desc2: np.ndarray) -> int:
    """
    Fast Hamming distance using lookup table.
    
    Args:
        desc1: First descriptor
        desc2: Second descriptor
        
    Returns:
        Hamming distance
    """
    # Precomputed popcount for bytes 0-255
    POPCOUNT = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)
    
    xor_result = desc1 ^ desc2
    return int(np.sum(POPCOUNT[xor_result]))


def brute_force_match(desc1: np.ndarray, desc2: np.ndarray) -> List[Match]:
    """
    Brute-force matching: match each descriptor in desc1 to all in desc2.
    
    For each descriptor in the query set, find the best (closest) match
    in the training set.
    
    Args:
        desc1: Query descriptors (N1 x D)
        desc2: Training descriptors (N2 x D)
        
    Returns:
        List of Match objects (one per query descriptor)
    """
    if len(desc1) == 0 or len(desc2) == 0:
        return []
    
    matches = []
    
    for i in range(len(desc1)):
        best_j = -1
        best_dist = float('inf')
        
        for j in range(len(desc2)):
            dist = hamming_distance(desc1[i], desc2[j])
            if dist < best_dist:
                best_dist = dist
                best_j = j
        
        if best_j >= 0:
            matches.append(Match(i, best_j, int(best_dist)))
    
    return matches


def brute_force_match_knn(desc1: np.ndarray, desc2: np.ndarray, 
                          k: int = 2) -> List[List[Match]]:
    """
    K-nearest-neighbor matching.
    
    For each descriptor, find the k best matches. This is useful for
    ratio test filtering.
    
    Args:
        desc1: Query descriptors
        desc2: Training descriptors
        k: Number of neighbors
        
    Returns:
        List of lists, each containing k matches for a query descriptor
    """
    if len(desc1) == 0 or len(desc2) == 0:
        return []
    
    all_matches = []
    
    for i in range(len(desc1)):
        # Compute all distances
        distances = []
        for j in range(len(desc2)):
            dist = hamming_distance(desc1[i], desc2[j])
            distances.append((j, dist))
        
        # Sort by distance
        distances.sort(key=lambda x: x[1])
        
        # Take top k
        knn = [Match(i, j, int(d)) for j, d in distances[:k]]
        all_matches.append(knn)
    
    return all_matches


def ratio_test(knn_matches: List[List[Match]], 
               ratio: float = 0.75) -> List[Match]:
    """
    Lowe's ratio test for filtering matches.
    
    A match is kept only if the distance to the best match is significantly
    smaller than the distance to the second best. This filters out ambiguous
    matches where the descriptor could match multiple targets equally well.
    
    Args:
        knn_matches: List of k-nearest matches for each query descriptor
        ratio: Maximum ratio of best to second-best distance
        
    Returns:
        Filtered list of good matches
    """
    good_matches = []
    
    for matches in knn_matches:
        if len(matches) < 2:
            continue
        
        best = matches[0]
        second_best = matches[1]
        
        # Avoid division by zero
        if second_best.distance == 0:
            continue
        
        # Apply ratio test
        if best.distance / second_best.distance < ratio:
            good_matches.append(best)
    
    return good_matches


def cross_check_matches(matches_1to2: List[Match], 
                        matches_2to1: List[Match]) -> List[Match]:
    """
    Cross-check matching for more robust results.
    
    A match (i, j) is kept only if:
    - j is the best match for i in image 2, AND
    - i is the best match for j in image 1
    
    Args:
        matches_1to2: Matches from image 1 to 2
        matches_2to1: Matches from image 2 to 1
        
    Returns:
        Cross-checked matches
    """
    # Build lookup from matches_2to1
    reverse_lookup = {}
    for m in matches_2to1:
        reverse_lookup[m.queryIdx] = m.trainIdx
    
    # Check each match in 1->2
    good_matches = []
    for m in matches_1to2:
        # m.queryIdx is in image 1, m.trainIdx is in image 2
        # Check if matching backwards gives us the original point
        if m.trainIdx in reverse_lookup:
            if reverse_lookup[m.trainIdx] == m.queryIdx:
                good_matches.append(m)
    
    return good_matches


def match_features(kp1: List[Keypoint], desc1: np.ndarray,
                   kp2: List[Keypoint], desc2: np.ndarray,
                   ratio_threshold: float = 0.75,
                   cross_check: bool = True) -> Tuple[List[Keypoint], List[Keypoint], List[Match]]:
    """
    Complete feature matching pipeline.
    
    1. Compute k-NN matches (k=2)
    2. Apply ratio test
    3. Optionally apply cross-check
    
    Args:
        kp1: Keypoints from image 1
        desc1: Descriptors from image 1
        kp2: Keypoints from image 2
        desc2: Descriptors from image 2
        ratio_threshold: For ratio test
        cross_check: Whether to apply cross-check
        
    Returns:
        (matched_kp1, matched_kp2, matches)
    """
    # k-NN matching
    knn_1to2 = brute_force_match_knn(desc1, desc2, k=2)
    
    # Ratio test
    matches = ratio_test(knn_1to2, ratio=ratio_threshold)
    
    # Cross-check if requested
    if cross_check and len(matches) > 0:
        knn_2to1 = brute_force_match_knn(desc2, desc1, k=1)
        matches_2to1 = [m[0] for m in knn_2to1 if len(m) > 0]
        matches = cross_check_matches(matches, matches_2to1)
    
    # Extract matched keypoints
    matched_kp1 = [kp1[m.queryIdx] for m in matches]
    matched_kp2 = [kp2[m.trainIdx] for m in matches]
    
    return matched_kp1, matched_kp2, matches


def get_matched_points(kp1: List[Keypoint], kp2: List[Keypoint],
                       matches: List[Match]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract matched point coordinates.
    
    Args:
        kp1: Keypoints from image 1
        kp2: Keypoints from image 2
        matches: List of matches
        
    Returns:
        (points1, points2) as Nx2 arrays
    """
    pts1 = np.array([[kp1[m.queryIdx].x, kp1[m.queryIdx].y] for m in matches])
    pts2 = np.array([[kp2[m.trainIdx].x, kp2[m.trainIdx].y] for m in matches])
    return pts1, pts2


def visualize_matches(img1: np.ndarray, img2: np.ndarray,
                      kp1: List[Keypoint], kp2: List[Keypoint],
                      matches: List[Match],
                      max_matches: int = 50) -> np.ndarray:
    """
    Visualize matches between two images.
    
    Creates a side-by-side image with lines connecting matched keypoints.
    
    Args:
        img1: First image
        img2: Second image
        kp1: Keypoints in first image
        kp2: Keypoints in second image
        matches: List of matches
        max_matches: Maximum matches to draw (for clarity)
        
    Returns:
        Visualization image
    """
    # Ensure same height
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    max_h = max(h1, h2)
    
    # Ensure RGB
    if len(img1.shape) == 2:
        img1 = np.stack([img1] * 3, axis=-1)
    if len(img2.shape) == 2:
        img2 = np.stack([img2] * 3, axis=-1)
    
    # Ensure float in [0, 1]
    if img1.max() > 1:
        img1 = img1.astype(np.float64) / 255.0
    if img2.max() > 1:
        img2 = img2.astype(np.float64) / 255.0
    
    # Create side-by-side image
    vis = np.zeros((max_h, w1 + w2, 3), dtype=np.float64)
    vis[:h1, :w1] = img1
    vis[:h2, w1:w1+w2] = img2
    
    # Sort matches by distance and take top ones
    sorted_matches = sorted(matches, key=lambda m: m.distance)[:max_matches]
    
    # Draw matches
    colors = np.random.rand(len(sorted_matches), 3)
    
    for i, m in enumerate(sorted_matches):
        pt1 = kp1[m.queryIdx]
        pt2 = kp2[m.trainIdx]
        
        x1, y1 = int(pt1.x), int(pt1.y)
        x2, y2 = int(pt2.x) + w1, int(pt2.y)  # Offset for second image
        
        color = colors[i]
        
        # Draw line
        steps = max(abs(x2 - x1), abs(y2 - y1), 1)
        for t in range(steps + 1):
            lx = int(x1 + t * (x2 - x1) / steps)
            ly = int(y1 + t * (y2 - y1) / steps)
            if 0 <= lx < vis.shape[1] and 0 <= ly < vis.shape[0]:
                vis[ly, lx] = color
        
        # Draw circles at endpoints
        for angle in np.linspace(0, 2 * np.pi, 12):
            cx1 = int(x1 + 3 * np.cos(angle))
            cy1 = int(y1 + 3 * np.sin(angle))
            cx2 = int(x2 + 3 * np.cos(angle))
            cy2 = int(y2 + 3 * np.sin(angle))
            
            if 0 <= cx1 < vis.shape[1] and 0 <= cy1 < vis.shape[0]:
                vis[cy1, cx1] = color
            if 0 <= cx2 < vis.shape[1] and 0 <= cy2 < vis.shape[0]:
                vis[cy2, cx2] = color
    
    return vis
