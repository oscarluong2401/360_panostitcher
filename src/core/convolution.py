"""
Convolution and filtering operations.
All implementations from scratch without scipy/cv2.
"""

import numpy as np
from typing import Tuple, Optional


def convolve2d(image: np.ndarray, kernel: np.ndarray, 
               padding: str = 'same') -> np.ndarray:
    """
    2D convolution implemented from scratch.
    
    Args:
        image: Input image (H, W) or (H, W, C)
        kernel: Convolution kernel (Kh, Kw)
        padding: 'same' to keep original size, 'valid' for no padding
        
    Returns:
        Convolved image
    """
    # Handle multi-channel images
    if len(image.shape) == 3:
        # Convolve each channel separately
        result = np.zeros_like(image)
        for c in range(image.shape[2]):
            result[:, :, c] = convolve2d(image[:, :, c], kernel, padding)
        return result
    
    h, w = image.shape
    kh, kw = kernel.shape
    
    # Padding size
    pad_h = kh // 2
    pad_w = kw // 2
    
    if padding == 'same':
        # Zero padding
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
        out_h, out_w = h, w
    else:  # valid
        padded = image
        out_h = h - kh + 1
        out_w = w - kw + 1
    
    # Output array
    output = np.zeros((out_h, out_w), dtype=np.float64)
    
    # Flip kernel for convolution (not correlation)
    kernel_flipped = np.flip(np.flip(kernel, 0), 1)
    
    # Perform convolution
    for i in range(out_h):
        for j in range(out_w):
            # Extract region
            region = padded[i:i+kh, j:j+kw]
            # Element-wise multiplication and sum
            output[i, j] = np.sum(region * kernel_flipped)
    
    return output


def convolve2d_separable(image: np.ndarray, 
                          kernel_x: np.ndarray, 
                          kernel_y: np.ndarray) -> np.ndarray:
    """
    Separable 2D convolution for efficiency.
    For a separable kernel K = ky * kx^T, we can do two 1D convolutions.
    
    Args:
        image: Input image (H, W)
        kernel_x: 1D horizontal kernel
        kernel_y: 1D vertical kernel
        
    Returns:
        Convolved image
    """
    # First convolve horizontally
    temp = convolve1d_horizontal(image, kernel_x)
    # Then convolve vertically
    result = convolve1d_vertical(temp, kernel_y)
    return result


def convolve1d_horizontal(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    1D horizontal convolution.
    
    Args:
        image: Input image (H, W)
        kernel: 1D kernel
        
    Returns:
        Convolved image
    """
    h, w = image.shape
    k_size = len(kernel)
    pad = k_size // 2
    
    # Pad horizontally
    padded = np.pad(image, ((0, 0), (pad, pad)), mode='reflect')
    
    output = np.zeros_like(image)
    
    for j in range(w):
        output[:, j] = np.sum(padded[:, j:j+k_size] * kernel, axis=1)
    
    return output


def convolve1d_vertical(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    1D vertical convolution.
    
    Args:
        image: Input image (H, W)
        kernel: 1D kernel
        
    Returns:
        Convolved image
    """
    h, w = image.shape
    k_size = len(kernel)
    pad = k_size // 2
    
    # Pad vertically
    padded = np.pad(image, ((pad, pad), (0, 0)), mode='reflect')
    
    output = np.zeros_like(image)
    
    for i in range(h):
        output[i, :] = np.sum(padded[i:i+k_size, :] * kernel.reshape(-1, 1), axis=0)
    
    return output


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """
    Generate a 2D Gaussian kernel.
    
    Args:
        size: Kernel size (will be made odd if even)
        sigma: Standard deviation of Gaussian
        
    Returns:
        2D Gaussian kernel (normalized to sum to 1)
    """
    if size % 2 == 0:
        size += 1
    
    # Create coordinate grid centered at 0
    half = size // 2
    x, y = np.mgrid[-half:half+1, -half:half+1]
    
    # Gaussian formula
    g = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    
    # Normalize
    return g / g.sum()


def gaussian_kernel_1d(size: int, sigma: float) -> np.ndarray:
    """
    Generate a 1D Gaussian kernel for separable convolution.
    
    Args:
        size: Kernel size
        sigma: Standard deviation
        
    Returns:
        1D Gaussian kernel (normalized)
    """
    if size % 2 == 0:
        size += 1
    
    half = size // 2
    x = np.arange(-half, half + 1)
    
    g = np.exp(-x**2 / (2 * sigma**2))
    return g / g.sum()


def gaussian_blur(image: np.ndarray, sigma: float, 
                  size: Optional[int] = None) -> np.ndarray:
    """
    Apply Gaussian blur to image.
    
    Args:
        image: Input image
        sigma: Standard deviation of Gaussian
        size: Kernel size (auto-computed if None)
        
    Returns:
        Blurred image
    """
    # Auto-compute kernel size (rule of thumb: 6*sigma)
    if size is None:
        size = int(6 * sigma) + 1
        if size % 2 == 0:
            size += 1
    
    # Use separable convolution for efficiency
    kernel_1d = gaussian_kernel_1d(size, sigma)
    
    # Handle multi-channel
    if len(image.shape) == 3:
        result = np.zeros_like(image)
        for c in range(image.shape[2]):
            result[:, :, c] = convolve2d_separable(
                image[:, :, c], kernel_1d, kernel_1d
            )
        return result
    
    return convolve2d_separable(image, kernel_1d, kernel_1d)


def sobel_x_kernel() -> np.ndarray:
    """Get Sobel kernel for horizontal gradient."""
    return np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float64)


def sobel_y_kernel() -> np.ndarray:
    """Get Sobel kernel for vertical gradient."""
    return np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ], dtype=np.float64)


def sobel_gradients(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute image gradients using Sobel operators.
    
    Args:
        image: Input grayscale image
        
    Returns:
        (Ix, Iy) - Horizontal and vertical gradients
    """
    Ix = convolve2d(image, sobel_x_kernel())
    Iy = convolve2d(image, sobel_y_kernel())
    return Ix, Iy


def gradient_magnitude(Ix: np.ndarray, Iy: np.ndarray) -> np.ndarray:
    """
    Compute gradient magnitude from x and y gradients.
    
    Args:
        Ix: Horizontal gradient
        Iy: Vertical gradient
        
    Returns:
        Gradient magnitude
    """
    return np.sqrt(Ix**2 + Iy**2)


def gradient_direction(Ix: np.ndarray, Iy: np.ndarray) -> np.ndarray:
    """
    Compute gradient direction from x and y gradients.
    
    Args:
        Ix: Horizontal gradient
        Iy: Vertical gradient
        
    Returns:
        Gradient direction in radians [-pi, pi]
    """
    return np.arctan2(Iy, Ix)


def image_pyramid(image: np.ndarray, levels: int, 
                  scale: float = 0.5) -> list:
    """
    Build image pyramid (successive downsampling).
    
    Args:
        image: Input image
        levels: Number of pyramid levels
        scale: Scale factor between levels (typically 0.5)
        
    Returns:
        List of images at different scales (largest first)
    """
    pyramid = [image]
    
    for _ in range(levels - 1):
        # Blur before downsampling (anti-aliasing)
        blurred = gaussian_blur(pyramid[-1], sigma=1.0)
        
        # Downsample
        h, w = blurred.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        
        if new_h < 2 or new_w < 2:
            break
            
        # Simple downsampling by striding
        if len(blurred.shape) == 3:
            downsampled = blurred[::2, ::2, :]
        else:
            downsampled = blurred[::2, ::2]
        
        pyramid.append(downsampled)
    
    return pyramid


def laplacian_of_gaussian(image: np.ndarray, sigma: float) -> np.ndarray:
    """
    Compute Laplacian of Gaussian (blob detector).
    
    Args:
        image: Input grayscale image
        sigma: Gaussian sigma
        
    Returns:
        LoG response
    """
    # Apply Gaussian blur
    blurred = gaussian_blur(image, sigma)
    
    # Laplacian kernel
    laplacian_kernel = np.array([
        [0,  1, 0],
        [1, -4, 1],
        [0,  1, 0]
    ], dtype=np.float64)
    
    # Apply Laplacian
    log = convolve2d(blurred, laplacian_kernel)
    
    # Normalize by sigma^2 for scale invariance
    return log * (sigma ** 2)


def sharpen(image: np.ndarray, amount: float = 1.0) -> np.ndarray:
    """
    Sharpen image using unsharp masking.
    
    Args:
        image: Input image
        amount: Sharpening amount (1.0 = standard)
        
    Returns:
        Sharpened image
    """
    # Blur the image
    blurred = gaussian_blur(image, sigma=1.0)
    
    # Unsharp mask: original + amount * (original - blurred)
    sharpened = image + amount * (image - blurred)
    
    # Clip to valid range
    return np.clip(sharpened, 0, 1)
