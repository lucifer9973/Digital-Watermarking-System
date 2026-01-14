import numpy as np
from skimage.metrics import structural_similarity as ssim
from typing import Dict, Union
import logging
import warnings

# Suppress overflow warnings in NCC calculation (we handle them explicitly)
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*overflow.*')

class MetricsCalculator:
    """
    Calculates quality metrics for watermarked images and extracted watermarks
    Metrics: PSNR, MSE, SSIM, NCC
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def calculate_psnr(self, original: np.ndarray, modified: np.ndarray) -> float:
        """Calculate Peak Signal-to-Noise Ratio"""
        # Ensure inputs are numpy arrays, not tuples
        if isinstance(original, tuple) or isinstance(modified, tuple):
            raise ValueError("PSNR calculation: Input cannot be a tuple")
        if not isinstance(original, np.ndarray):
            original = np.asarray(original)
        if not isinstance(modified, np.ndarray):
            modified = np.asarray(modified)
        
        # Ensure both arrays have the same dtype for calculation
        orig = original.astype(np.float64)
        mod = modified.astype(np.float64)
        
        mse = np.mean((orig - mod) ** 2)
        if mse == 0:
            return float('inf')
        
        # Determine max pixel value based on dtype
        if original.dtype == np.uint8 or modified.dtype == np.uint8:
            max_pixel = 255.0
        else:
            max_pixel = max(original.max(), modified.max())
            if max_pixel <= 0:
                max_pixel = 1.0
        
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        return psnr if np.isfinite(psnr) else 0.0
    
    def calculate_mse(self, original: np.ndarray, modified: np.ndarray) -> float:
        """Calculate Mean Square Error"""
        # Ensure inputs are numpy arrays, not tuples
        if isinstance(original, tuple) or isinstance(modified, tuple):
            raise ValueError("MSE calculation: Input cannot be a tuple")
        if not isinstance(original, np.ndarray):
            original = np.asarray(original)
        if not isinstance(modified, np.ndarray):
            modified = np.asarray(modified)
        return np.mean((original - modified) ** 2)
    
    def calculate_ncc(self, original: np.ndarray, modified: np.ndarray) -> float:
        """Calculate Normalized Cross-Correlation"""
        # Ensure inputs are numpy arrays, not tuples
        if isinstance(original, tuple) or isinstance(modified, tuple):
            raise ValueError("NCC calculation: Input cannot be a tuple")
        if not isinstance(original, np.ndarray):
            original = np.asarray(original)
        if not isinstance(modified, np.ndarray):
            modified = np.asarray(modified)
        
        # Convert to float64 to avoid overflow
        original = original.astype(np.float64)
        modified = modified.astype(np.float64)
        
        # Use more numerically stable calculation to avoid overflow
        # Instead of: sum(orig * mod) / sqrt(sum(orig^2) * sum(mod^2))
        # Use: sum(orig * mod) / (sqrt(sum(orig^2)) * sqrt(sum(mod^2)))
        orig_sum_sq = np.sum(original * original)
        mod_sum_sq = np.sum(modified * modified)
        
        if orig_sum_sq == 0 or mod_sum_sq == 0:
            return 0.0
        
        # Check for potential overflow
        cross_corr = np.sum(original * modified)
        if not np.isfinite(cross_corr):
            return 0.0
        
        # Calculate NCC with overflow protection
        denominator = np.sqrt(orig_sum_sq) * np.sqrt(mod_sum_sq)
        if denominator == 0 or not np.isfinite(denominator):
            return 0.0
        
        ncc = cross_corr / denominator
        
        # Clamp to valid range [-1, 1] for correlation
        ncc = np.clip(ncc, -1.0, 1.0)
        
        return float(ncc) if np.isfinite(ncc) else 0.0
    
    def calculate_ssim(self, original: np.ndarray, modified: np.ndarray) -> float:
        """Calculate Structural Similarity Index"""
        # Ensure inputs are numpy arrays, not tuples
        if isinstance(original, tuple) or isinstance(modified, tuple):
            raise ValueError("SSIM calculation: Input cannot be a tuple")
        if not isinstance(original, np.ndarray):
            original = np.asarray(original)
        if not isinstance(modified, np.ndarray):
            modified = np.asarray(modified)
        # Determine data range based on image dtype
        if original.dtype == np.uint8 or modified.dtype == np.uint8:
            data_range = 255.0
        else:
            # For float images, use the actual range
            data_range = max(original.max() - original.min(), 
                           modified.max() - modified.min())
            if data_range == 0:
                data_range = 1.0
        
        try:
            return ssim(original, modified, data_range=data_range)
        except Exception as e:
            self.logger.warning(f"SSIM calculation failed: {str(e)}")
            return 0.0
    
    def calculate_all_metrics(self, 
                            original: np.ndarray, 
                            modified: np.ndarray) -> Dict[str, float]:
        """
        Calculate all metrics at once
        
        Args:
            original: Original image/watermark
            modified: Modified/extracted image/watermark
            
        Returns:
            Dictionary containing all metrics
        """
        # CRITICAL: Ensure inputs are numpy arrays, not tuples
        if isinstance(original, tuple):
            raise ValueError(f"original parameter is a tuple! Expected numpy array. Tuple contents: {[type(x) for x in original]}")
        if isinstance(modified, tuple):
            raise ValueError(f"modified parameter is a tuple! Expected numpy array. Tuple contents: {[type(x) for x in modified]}")
        
        # Convert to numpy arrays if needed
        if not isinstance(original, np.ndarray):
            original = np.asarray(original)
        if not isinstance(modified, np.ndarray):
            modified = np.asarray(modified)
        
        # Now safe to access shape
        if original.shape != modified.shape:
            raise ValueError(f"Images must have same dimensions. Original: {original.shape}, Modified: {modified.shape}")
            
        try:
            metrics = {
                'psnr': self.calculate_psnr(original, modified),
                'mse': self.calculate_mse(original, modified),
                'ncc': self.calculate_ncc(original, modified),
                'ssim': self.calculate_ssim(original, modified)
            }
            
            self.logger.info(f"Metrics calculated successfully: {metrics}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {str(e)}")
            raise