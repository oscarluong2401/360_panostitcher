import numpy as np

def compute_energy(overlap1: np.ndarray, overlap2: np.ndarray) -> np.ndarray:
    """
    Compute energy map (squared difference) between two overlapping image regions.
    Ideally, use gradient magnitude difference or color difference.
    """
    # Simple squared difference of luminance/intensity
    diff = overlap1.astype(np.float32) - overlap2.astype(np.float32)
    energy = np.sum(diff**2, axis=2) if len(diff.shape) == 3 else diff**2
    return energy

def find_optimal_seam(energy: np.ndarray) -> np.ndarray:
    """
    Find the vertical seam with minimum accumulated energy using Dynamic Programming.
    Returns a boolean mask where True indicates pixels belonging to the left image (left of seam).
    """
    h, w = energy.shape
    M = energy.copy()
    
    # Forward pass: accumulate energy
    # M[i, j] = energy[i, j] + min(M[i-1, j-1], M[i-1, j], M[i-1, j+1])
    
    # We can use vectorization to speed this up slightly, but iterating rows is easiest for the logic
    for i in range(1, h):
        # Edges need care
        row_prev = M[i-1]
        
        # Shifted versions for vectorization
        # up_left: M[i-1, j-1] -> shift right
        # up_right: M[i-1, j+1] -> shift left
        
        # However, simplistic loop is clear and fast enough for these strip sizes
        for j in range(w):
            min_prev = row_prev[j]
            if j > 0:
                min_prev = min(min_prev, row_prev[j-1])
            if j < w - 1:
                min_prev = min(min_prev, row_prev[j+1])
            
            M[i, j] += min_prev
            
    # Backward pass: trace seam
    mask = np.ones((h, w), dtype=bool)
    path = []
    
    # Start from bottom
    min_idx = np.argmin(M[-1])
    path.append(min_idx)
    mask[-1, min_idx+1:] = False # Right side of seam is False
    
    for i in range(h-2, -1, -1):
        j = path[-1]
        
        # Look at 3 neighbors
        start = max(0, j-1)
        end = min(w, j+2)
        
        # local slice
        candidates = M[i, start:end]
        
        # Offset calculation
        # if j=10, start=9. candidates are [9, 10, 11]
        # argmin returns 0, 1, or 2. 
        # new_j = start + argmin
        offset = np.argmin(candidates)
        next_j = start + offset
        path.append(next_j)
        
        mask[i, next_j+1:] = False
        
    return mask

def create_seam_mask(overlap1: np.ndarray, overlap2: np.ndarray) -> np.ndarray:
    """
    Create a binary mask for the overlap region using optimal seam finding.
    Returns mask for the LEFT image (1s on left of seam, 0s on right).
    """
    energy = compute_energy(overlap1, overlap2)
    mask = find_optimal_seam(energy)
    return mask
