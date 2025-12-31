"""
Harris corner detector implementation.
Detects corners by analyzing the local structure tensor.
"""

import numpy as np
from typing import List, Tuple
from ..core.convolution import gaussian_blur, sobel_gradients, convolve2d, gaussian_kernel


class Keypoint:
    """
    Represents a detected keypoint/feature point.
    """
    def __init__(self, x: float, y: float, response: float = 0.0, 
                 scale: float = 1.0, orientation: float = 0.0):
        self.x = x
        self.y = y
        self.response = response
        self.scale = scale
        self.orientation = orientation  # in radians
    
    @property
    def pt(self) -> Tuple[float, float]:
        """Get point as tuple."""
        return (self.x, self.y)
    
    def __repr__(self) -> str:
        return f"Keypoint(x={self.x:.1f}, y={self.y:.1f}, r={self.response:.4f})"


def harris_corners(image: np.ndarray, 
                   k: float = 0.04,
                   threshold: float = 0.01,
                   window_size: int = 3,
                   sigma: float = 1.0) -> List[Keypoint]:
    """
    Harris corner detection algorithm.
    
    The Harris detector finds corners by analyzing local image structure.
    A corner is a point where the image has significant gradient in 
    multiple directions, meaning a small shift in any direction causes
    a large change in appearance.
    
    Algorithm:
    1. Compute gradients Ix, Iy using Sobel
    2. Compute products: Ix², Iy², IxIy
    3. Apply Gaussian window to smooth products (structure tensor)
    4. For each pixel, compute Harris response:
       R = det(M) - k * trace(M)²
       where M is the structure tensor
    5. Threshold and return corner locations
    
    Args:
        image: Grayscale image (H, W) with values in [0, 1]
        k: Harris detector free parameter (typically 0.04-0.06)
        threshold: Response threshold as fraction of max response
        window_size: Size of Gaussian window for smoothing
        sigma: Gaussian sigma for smoothing
        
    Returns:
        List of Keypoint objects for detected corners
    """
    # Ensure grayscale
    if len(image.shape) == 3:
        # Convert to grayscale
        image = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    
    # Step 1: Compute image gradients
    Ix, Iy = sobel_gradients(image)
    
    # Step 2: Compute products of gradients at each pixel
    Ix2 = Ix ** 2
    Iy2 = Iy ** 2
    IxIy = Ix * Iy
    
    # Step 3: Apply Gaussian window to smooth the products
    # This creates a weighted sum over a local neighborhood
    Sx2 = gaussian_blur(Ix2, sigma=sigma)
    Sy2 = gaussian_blur(Iy2, sigma=sigma)
    Sxy = gaussian_blur(IxIy, sigma=sigma)
    
    # Step 4: Compute Harris response at each pixel
    # For the structure tensor M = [[Sx2, Sxy], [Sxy, Sy2]]:
    # det(M) = Sx2 * Sy2 - Sxy^2
    # trace(M) = Sx2 + Sy2
    # R = det(M) - k * trace(M)^2
    
    det_M = Sx2 * Sy2 - Sxy ** 2
    trace_M = Sx2 + Sy2
    
    response = det_M - k * (trace_M ** 2)
    
    # Step 5: Threshold response
    max_response = response.max()
    if max_response <= 0:
        return []
    
    threshold_value = threshold * max_response
    
    # Find pixels above threshold
    corners = []
    h, w = response.shape
    
    # Avoid edges (corners near image boundary are often unreliable)
    margin = max(window_size, 5)
    
    for y in range(margin, h - margin):
        for x in range(margin, w - margin):
            if response[y, x] > threshold_value:
                corners.append(Keypoint(
                    x=float(x),
                    y=float(y),
                    response=float(response[y, x])
                ))
    
    return corners


def harris_response_map(image: np.ndarray, k: float = 0.04, 
                        sigma: float = 1.0) -> np.ndarray:
    """
    Compute just the Harris response map (useful for visualization).
    
    Args:
        image: Grayscale image
        k: Harris parameter
        sigma: Gaussian sigma
        
    Returns:
        Harris response map
    """
    if len(image.shape) == 3:
        image = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    
    Ix, Iy = sobel_gradients(image)
    
    Sx2 = gaussian_blur(Ix ** 2, sigma=sigma)
    Sy2 = gaussian_blur(Iy ** 2, sigma=sigma)
    Sxy = gaussian_blur(Ix * Iy, sigma=sigma)
    
    det_M = Sx2 * Sy2 - Sxy ** 2
    trace_M = Sx2 + Sy2
    
    return det_M - k * (trace_M ** 2)


def shi_tomasi_corners(image: np.ndarray,
                       threshold: float = 0.01,
                       sigma: float = 1.0) -> List[Keypoint]:
    """
    Shi-Tomasi corner detector (Good Features to Track).
    
    Instead of Harris response R = det(M) - k*trace(M)^2,
    uses R = min(λ1, λ2) where λ1, λ2 are eigenvalues.
    
    This is often more stable than Harris.
    
    Args:
        image: Grayscale image
        threshold: Response threshold as fraction of max
        sigma: Gaussian sigma
        
    Returns:
        List of Keypoint objects
    """
    if len(image.shape) == 3:
        image = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    
    Ix, Iy = sobel_gradients(image)
    
    Sx2 = gaussian_blur(Ix ** 2, sigma=sigma)
    Sy2 = gaussian_blur(Iy ** 2, sigma=sigma)
    Sxy = gaussian_blur(Ix * Iy, sigma=sigma)
    
    # Compute eigenvalues of 2x2 matrix [[a, b], [b, c]]
    # λ = (a+c)/2 ± sqrt((a-c)²/4 + b²)
    a = Sx2
    b = Sxy
    c = Sy2
    
    trace_half = (a + c) / 2
    discriminant = np.sqrt(((a - c) / 2) ** 2 + b ** 2)
    
    lambda1 = trace_half + discriminant
    lambda2 = trace_half - discriminant
    
    # Shi-Tomasi response is minimum eigenvalue
    response = np.minimum(lambda1, lambda2)
    
    max_response = response.max()
    if max_response <= 0:
        return []
    
    threshold_value = threshold * max_response
    
    corners = []
    h, w = response.shape
    margin = 5
    
    for y in range(margin, h - margin):
        for x in range(margin, w - margin):
            if response[y, x] > threshold_value:
                corners.append(Keypoint(
                    x=float(x),
                    y=float(y),
                    response=float(response[y, x])
                ))
    
    return corners
