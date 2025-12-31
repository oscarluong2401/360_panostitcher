"""
Main panorama stitching pipeline.
Combines all components into a complete stitcher.
"""

import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path

from ..core.image import Image
from ..features.harris import harris_corners, Keypoint
from ..features.nms import non_maximum_suppression, adaptive_non_maximum_suppression
from ..features.descriptor import compute_descriptors, visualize_keypoints
from ..features.matching import match_features, get_matched_points, visualize_matches
from ..geometry.homography import compute_homography_dlt, apply_homography
from ..geometry.ransac import ransac_homography
from ..geometry.transform import (
    compute_output_bounds, 
    warp_perspective_fast, 
    compute_transformed_corners
)
from ..blending.alpha_blend import alpha_blend, simple_average_blend
from ..blending.multiband import multiband_blend_with_masks, exposure_compensate


class PanoramaStitcher:
    """
    Complete panorama stitching pipeline.
    
    Stitches multiple images together by:
    1. Detecting features in each image
    2. Matching features between adjacent images
    3. Estimating homographies using RANSAC
    4. Warping images to a common coordinate frame
    5. Blending overlapping regions
    """
    
    def __init__(self,
                 # Feature detection parameters
                 harris_k: float = 0.04,
                 harris_threshold: float = 0.01,
                 nms_radius: int = 10,
                 max_keypoints: int = 1000,
                 
                 # Matching parameters
                 ratio_threshold: float = 0.75,
                 use_cross_check: bool = True,
                 
                 # RANSAC parameters
                 ransac_iterations: int = 2000,
                 ransac_threshold: float = 4.0,
                 min_inliers: int = 10,
                 
                 # Blending parameters
                 blend_mode: str = 'multiband',
                 blend_levels: int = 5,
                 exposure_compensation: bool = True,
                 
                 # Debug
                 verbose: bool = True):
        """
        Initialize the stitcher with configuration.
        
        Args:
            harris_k: Harris detector free parameter
            harris_threshold: Corner response threshold
            nms_radius: Non-maximum suppression radius
            max_keypoints: Maximum keypoints per image
            ratio_threshold: Lowe's ratio test threshold
            use_cross_check: Whether to use cross-check matching
            ransac_iterations: RANSAC iterations
            ransac_threshold: RANSAC inlier threshold
            min_inliers: Minimum inliers for valid match
            blend_mode: 'multiband', 'alpha', or 'average'
            blend_levels: Pyramid levels for multiband blend
            exposure_compensation: Whether to compensate exposure
            verbose: Print progress info
        """
        self.harris_k = harris_k
        self.harris_threshold = harris_threshold
        self.nms_radius = nms_radius
        self.max_keypoints = max_keypoints
        
        self.ratio_threshold = ratio_threshold
        self.use_cross_check = use_cross_check
        
        self.ransac_iterations = ransac_iterations
        self.ransac_threshold = ransac_threshold
        self.min_inliers = min_inliers
        
        self.blend_mode = blend_mode
        self.blend_levels = blend_levels
        self.exposure_compensation = exposure_compensation
        
        self.verbose = verbose
        
        # Storage for intermediate results
        self._keypoints = []
        self._descriptors = []
        self._matches = []
        self._homographies = []
    
    def log(self, message: str):
        """Print log message if verbose."""
        if self.verbose:
            print(f"[Stitcher] {message}")
    
    def detect_features(self, image: np.ndarray) -> Tuple[List[Keypoint], np.ndarray]:
        """
        Detect and describe features in an image.
        
        Args:
            image: Input image
            
        Returns:
            (keypoints, descriptors)
        """
        self.log("Detecting corners...")
        
        # Detect corners
        keypoints = harris_corners(
            image, 
            k=self.harris_k,
            threshold=self.harris_threshold
        )
        
        self.log(f"  Found {len(keypoints)} corners")
        
        # Non-maximum suppression
        keypoints = non_maximum_suppression(
            keypoints,
            radius=self.nms_radius,
            max_keypoints=self.max_keypoints
        )
        
        self.log(f"  After NMS: {len(keypoints)} keypoints")
        
        # Compute descriptors
        keypoints, descriptors = compute_descriptors(image, keypoints)
        
        self.log(f"  Computed {len(keypoints)} descriptors")
        
        return keypoints, descriptors
    
    def match_images(self,
                     kp1: List[Keypoint], desc1: np.ndarray,
                     kp2: List[Keypoint], desc2: np.ndarray
                     ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Match features between two images.
        
        Args:
            kp1, desc1: Keypoints and descriptors from image 1
            kp2, desc2: Keypoints and descriptors from image 2
            
        Returns:
            (matched_pts1, matched_pts2, n_matches)
        """
        self.log("Matching features...")
        
        matched_kp1, matched_kp2, matches = match_features(
            kp1, desc1, kp2, desc2,
            ratio_threshold=self.ratio_threshold,
            cross_check=self.use_cross_check
        )
        
        self.log(f"  Found {len(matches)} matches")
        
        if len(matches) < 4:
            return np.array([]), np.array([]), 0
        
        pts1, pts2 = get_matched_points(kp1, kp2, matches)
        
        return pts1, pts2, len(matches)
    
    def estimate_homography(self,
                            pts1: np.ndarray,
                            pts2: np.ndarray) -> Tuple[Optional[np.ndarray], int]:
        """
        Estimate homography using RANSAC.
        
        Args:
            pts1: Points from image 1
            pts2: Points from image 2
            
        Returns:
            (homography, n_inliers)
        """
        if len(pts1) < 4:
            return None, 0
        
        self.log("Estimating homography with RANSAC...")
        
        H, inlier_mask = ransac_homography(
            pts1, pts2,
            n_iterations=self.ransac_iterations,
            threshold=self.ransac_threshold,
            min_inliers=self.min_inliers
        )
        
        n_inliers = np.sum(inlier_mask) if inlier_mask is not None else 0
        
        if H is not None:
            self.log(f"  Homography found with {n_inliers} inliers")
        else:
            self.log("  Failed to find valid homography")
        
        return H, n_inliers
    
    def blend_images(self,
                     img1: np.ndarray,
                     img2: np.ndarray,
                     mask1: np.ndarray,
                     mask2: np.ndarray) -> np.ndarray:
        """
        Blend two images together.
        
        Args:
            img1: First image
            img2: Second image (warped)
            mask1: Valid pixels in img1
            mask2: Valid pixels in img2
            
        Returns:
            Blended image
        """
        self.log(f"Blending images using {self.blend_mode} mode...")
        
        # Exposure compensation
        if self.exposure_compensation:
            img1, img2 = exposure_compensate(img1, img2, mask1, mask2)
        
        if self.blend_mode == 'multiband':
            blended = multiband_blend_with_masks(
                img1, img2, mask1, mask2, 
                levels=self.blend_levels
            )
        elif self.blend_mode == 'alpha':
            blended = alpha_blend(img1, img2, mask1, mask2, feather=True)
        else:  # average
            blended = simple_average_blend(img1, img2, mask1, mask2)
        
        return blended
    
    def stitch_pair(self, 
                    img1: np.ndarray, 
                    img2: np.ndarray) -> Optional[np.ndarray]:
        """
        Stitch two images together.
        
        Image 1 is the reference. Image 2 is warped to align with it.
        
        Args:
            img1: Reference image
            img2: Image to warp and stitch
            
        Returns:
            Stitched panorama (or None if failed)
        """
        self.log("=" * 50)
        self.log("Stitching image pair")
        self.log("=" * 50)
        
        # Step 1: Detect features
        kp1, desc1 = self.detect_features(img1)
        kp2, desc2 = self.detect_features(img2)
        
        if len(kp1) < 10 or len(kp2) < 10:
            self.log("ERROR: Not enough features detected")
            return None
        
        # Step 2: Match features
        pts1, pts2, n_matches = self.match_images(kp1, desc1, kp2, desc2)
        
        if n_matches < self.min_inliers:
            self.log("ERROR: Not enough matches found")
            return None
        
        # Step 3: Estimate homography (pts2 -> pts1, so img2 maps to img1's frame)
        H, n_inliers = self.estimate_homography(pts2, pts1)
        
        if H is None:
            self.log("ERROR: Could not estimate homography")
            return None
        
        # Step 4: Compute output bounds
        offset, output_shape = compute_output_bounds(
            H, img2.shape[:2], img1.shape[:2]
        )
        
        self.log(f"Output size: {output_shape[1]}x{output_shape[0]}")
        
        # Step 5: Create output canvas and place img1
        h_out, w_out = output_shape
        
        if len(img1.shape) == 3:
            canvas = np.zeros((h_out, w_out, img1.shape[2]), dtype=np.float64)
        else:
            canvas = np.zeros((h_out, w_out), dtype=np.float64)
        
        mask1 = np.zeros((h_out, w_out), dtype=bool)
        
        # Place img1 with offset
        x_off = int(offset[0])
        y_off = int(offset[1])
        h1, w1 = img1.shape[:2]
        
        y_start = max(0, y_off)
        y_end = min(h_out, y_off + h1)
        x_start = max(0, x_off)
        x_end = min(w_out, x_off + w1)
        
        src_y_start = max(0, -y_off)
        src_x_start = max(0, -x_off)
        src_y_end = src_y_start + (y_end - y_start)
        src_x_end = src_x_start + (x_end - x_start)
        
        canvas[y_start:y_end, x_start:x_end] = img1[src_y_start:src_y_end, src_x_start:src_x_end]
        mask1[y_start:y_end, x_start:x_end] = True
        
        # Step 6: Warp img2
        self.log("Warping image...")
        warped2, mask2 = warp_perspective_fast(img2, H, output_shape, offset)
        
        # Step 7: Blend
        result = self.blend_images(canvas, warped2, mask1, mask2)
        
        self.log("Stitching complete!")
        
        return result
    
    def stitch_multiple(self, images: List[np.ndarray]) -> Optional[np.ndarray]:
        """
        Stitch multiple images into a panorama.
        
        Uses the middle image as reference and stitches outward.
        
        Args:
            images: List of images (left to right order)
            
        Returns:
            Stitched panorama
        """
        n = len(images)
        
        if n == 0:
            return None
        if n == 1:
            return images[0]
        if n == 2:
            return self.stitch_pair(images[0], images[1])
        
        self.log(f"Stitching {n} images")
        
        # Use middle image as reference
        mid = n // 2
        result = images[mid].copy()
        
        # Stitch images to the left of center
        for i in range(mid - 1, -1, -1):
            self.log(f"\nStitching image {i} (left side)")
            result = self.stitch_pair(result, images[i])
            if result is None:
                self.log(f"Failed at image {i}")
                return None
        
        # Stitch images to the right of center
        for i in range(mid + 1, n):
            self.log(f"\nStitching image {i} (right side)")
            result = self.stitch_pair(result, images[i])
            if result is None:
                self.log(f"Failed at image {i}")
                return None
        
        return result
    
    def stitch_from_paths(self, paths: List[str]) -> Optional[np.ndarray]:
        """
        Stitch images from file paths.
        
        Args:
            paths: List of image paths
            
        Returns:
            Stitched panorama
        """
        images = []
        for path in paths:
            self.log(f"Loading: {path}")
            img = Image.load(path)
            images.append(img.data)
        
        return self.stitch_multiple(images)
    
    def visualize_features(self, image: np.ndarray) -> np.ndarray:
        """
        Visualize detected features on an image.
        
        Args:
            image: Input image
            
        Returns:
            Image with features drawn
        """
        kp, _ = self.detect_features(image)
        return visualize_keypoints(image, kp)
    
    def visualize_pair_matches(self,
                               img1: np.ndarray,
                               img2: np.ndarray) -> np.ndarray:
        """
        Visualize feature matches between two images.
        
        Args:
            img1: First image
            img2: Second image
            
        Returns:
            Visualization of matches
        """
        kp1, desc1 = self.detect_features(img1)
        kp2, desc2 = self.detect_features(img2)
        
        _, _, matches = match_features(
            kp1, desc1, kp2, desc2,
            ratio_threshold=self.ratio_threshold,
            cross_check=self.use_cross_check
        )
        
        return visualize_matches(img1, img2, kp1, kp2, matches)


def create_test_images() -> Tuple[np.ndarray, np.ndarray]:
    """
    Create simple test images for debugging.
    
    Returns:
        (img1, img2) - Two overlapping test images
    """
    h, w = 300, 400
    
    # Create gradient + patterns
    img1 = np.zeros((h, w, 3), dtype=np.float64)
    img2 = np.zeros((h, w, 3), dtype=np.float64)
    
    # Horizontal gradient
    for x in range(w):
        img1[:, x, 0] = x / w  # Red gradient
    
    for x in range(w):
        img2[:, x, 2] = x / w  # Blue gradient
    
    # Add some features (rectangles)
    img1[50:100, 50:100] = [1, 1, 1]
    img1[150:200, 100:150] = [0.5, 0.5, 0.5]
    img1[100:120, 200:250] = [1, 0, 0]
    
    # Shifted version for img2 (simulating overlap)
    img2[50:100, 150:200] = [1, 1, 1]  # Same rectangle, shifted right
    img2[150:200, 200:250] = [0.5, 0.5, 0.5]
    img2[100:120, 300:350] = [0, 0, 1]
    
    return img1, img2
