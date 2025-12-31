"""
Image class and I/O operations.
Uses only PIL for loading/saving - no OpenCV.
"""

import numpy as np
from PIL import Image as PILImage
from pathlib import Path
from typing import Union, Tuple, Optional


class Image:
    """
    Image container class with basic operations.
    All image data is stored as numpy arrays.
    """
    
    def __init__(self, data: np.ndarray):
        """
        Initialize Image with numpy array.
        
        Args:
            data: Image data as numpy array. 
                  Shape can be (H, W) for grayscale or (H, W, 3) for RGB.
                  Values should be in range [0, 255] or [0.0, 1.0].
        """
        self._data = data.astype(np.float64)
        
        # Normalize to [0, 1] if needed
        if self._data.max() > 1.0:
            self._data = self._data / 255.0
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'Image':
        """
        Load image from file.
        
        Args:
            path: Path to image file
            
        Returns:
            Image object
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        
        # Load with PIL
        pil_image = PILImage.open(path)
        
        # Convert to RGB if needed
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Convert to numpy array
        data = np.array(pil_image, dtype=np.float64) / 255.0
        
        return cls(data)
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save image to file.
        
        Args:
            path: Output path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to uint8
        data_uint8 = (self._data * 255).clip(0, 255).astype(np.uint8)
        
        # Handle grayscale
        if len(data_uint8.shape) == 2:
            pil_image = PILImage.fromarray(data_uint8, mode='L')
        else:
            pil_image = PILImage.fromarray(data_uint8, mode='RGB')
        
        pil_image.save(path)
    
    @property
    def data(self) -> np.ndarray:
        """Get image data as numpy array (values in [0, 1])."""
        return self._data
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Get image shape (H, W) or (H, W, C)."""
        return self._data.shape
    
    @property
    def height(self) -> int:
        """Get image height."""
        return self._data.shape[0]
    
    @property
    def width(self) -> int:
        """Get image width."""
        return self._data.shape[1]
    
    @property
    def channels(self) -> int:
        """Get number of channels (1 for grayscale, 3 for RGB)."""
        if len(self._data.shape) == 2:
            return 1
        return self._data.shape[2]
    
    def is_grayscale(self) -> bool:
        """Check if image is grayscale."""
        return len(self._data.shape) == 2
    
    def to_grayscale(self) -> 'Image':
        """
        Convert to grayscale using luminosity method.
        Y = 0.299*R + 0.587*G + 0.114*B
        
        Returns:
            Grayscale Image
        """
        if self.is_grayscale():
            return Image(self._data.copy())
        
        # Luminosity method (matches human perception)
        gray = (0.299 * self._data[:, :, 0] + 
                0.587 * self._data[:, :, 1] + 
                0.114 * self._data[:, :, 2])
        
        return Image(gray)
    
    def to_rgb(self) -> 'Image':
        """
        Convert grayscale to RGB by replicating channels.
        
        Returns:
            RGB Image
        """
        if not self.is_grayscale():
            return Image(self._data.copy())
        
        rgb = np.stack([self._data] * 3, axis=-1)
        return Image(rgb)
    
    def resize(self, new_height: int, new_width: int) -> 'Image':
        """
        Resize image using bilinear interpolation (implemented from scratch).
        
        Args:
            new_height: Target height
            new_width: Target width
            
        Returns:
            Resized Image
        """
        old_h, old_w = self.height, self.width
        
        # Create output array
        if self.is_grayscale():
            output = np.zeros((new_height, new_width), dtype=np.float64)
        else:
            output = np.zeros((new_height, new_width, 3), dtype=np.float64)
        
        # Scale factors
        scale_y = old_h / new_height
        scale_x = old_w / new_width
        
        # For each output pixel, find corresponding source location
        for y in range(new_height):
            for x in range(new_width):
                # Map to source coordinates (center of pixel)
                src_y = (y + 0.5) * scale_y - 0.5
                src_x = (x + 0.5) * scale_x - 0.5
                
                # Bilinear interpolation
                output[y, x] = self._bilinear_sample(src_x, src_y)
        
        return Image(output)
    
    def _bilinear_sample(self, x: float, y: float) -> np.ndarray:
        """
        Sample image at non-integer location using bilinear interpolation.
        
        Args:
            x: X coordinate (can be non-integer)
            y: Y coordinate (can be non-integer)
            
        Returns:
            Interpolated pixel value
        """
        h, w = self.height, self.width
        
        # Get integer coordinates
        x0 = int(np.floor(x))
        y0 = int(np.floor(y))
        x1 = x0 + 1
        y1 = y0 + 1
        
        # Clamp to valid range
        x0 = max(0, min(x0, w - 1))
        x1 = max(0, min(x1, w - 1))
        y0 = max(0, min(y0, h - 1))
        y1 = max(0, min(y1, h - 1))
        
        # Interpolation weights
        wx = x - np.floor(x)
        wy = y - np.floor(y)
        
        # Get four neighbor pixels
        p00 = self._data[y0, x0]
        p01 = self._data[y0, x1]
        p10 = self._data[y1, x0]
        p11 = self._data[y1, x1]
        
        # Bilinear interpolation
        return (p00 * (1 - wx) * (1 - wy) +
                p01 * wx * (1 - wy) +
                p10 * (1 - wx) * wy +
                p11 * wx * wy)
    
    def scale(self, factor: float) -> 'Image':
        """
        Scale image by a factor.
        
        Args:
            factor: Scale factor (e.g., 0.5 for half size)
            
        Returns:
            Scaled Image
        """
        new_h = int(self.height * factor)
        new_w = int(self.width * factor)
        return self.resize(new_h, new_w)
    
    def crop(self, x: int, y: int, width: int, height: int) -> 'Image':
        """
        Crop a region from the image.
        
        Args:
            x: Left coordinate
            y: Top coordinate
            width: Crop width
            height: Crop height
            
        Returns:
            Cropped Image
        """
        # Clamp to valid range
        x = max(0, min(x, self.width - 1))
        y = max(0, min(y, self.height - 1))
        x2 = min(x + width, self.width)
        y2 = min(y + height, self.height)
        
        return Image(self._data[y:y2, x:x2].copy())
    
    def pad(self, top: int, right: int, bottom: int, left: int, 
            value: float = 0.0) -> 'Image':
        """
        Pad image with constant value.
        
        Args:
            top: Top padding
            right: Right padding
            bottom: Bottom padding
            left: Left padding
            value: Padding value (0-1)
            
        Returns:
            Padded Image
        """
        if self.is_grayscale():
            padded = np.full(
                (self.height + top + bottom, self.width + left + right),
                value, dtype=np.float64
            )
            padded[top:top+self.height, left:left+self.width] = self._data
        else:
            padded = np.full(
                (self.height + top + bottom, self.width + left + right, 3),
                value, dtype=np.float64
            )
            padded[top:top+self.height, left:left+self.width, :] = self._data
        
        return Image(padded)
    
    def copy(self) -> 'Image':
        """Create a deep copy of the image."""
        return Image(self._data.copy())
    
    def as_uint8(self) -> np.ndarray:
        """Get image data as uint8 array (values in [0, 255])."""
        return (self._data * 255).clip(0, 255).astype(np.uint8)
    
    def __repr__(self) -> str:
        mode = "grayscale" if self.is_grayscale() else "RGB"
        return f"Image({self.width}x{self.height}, {mode})"
