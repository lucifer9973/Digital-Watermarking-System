import numpy as np
from typing import Dict, Optional
import logging
import cv2
from .dwt_transform import DWTTransform
from .svd_transform import SVDTransform
from ..utils.metrics import MetricsCalculator

class WatermarkExtractor:
    """
    Extracts watermark from watermarked image using original cover image and alpha.
    Includes robust denoising to clean up quantization artifacts.
    """
    
    def __init__(self, dwt_params: Optional[Dict] = None):
        self.dwt = DWTTransform(**(dwt_params or {}))
        self.svd = SVDTransform()
        self.metrics = MetricsCalculator()
        self.logger = logging.getLogger(__name__)

    def extract(self, 
                watermarked_image: np.ndarray,
                original_image: np.ndarray,
                alpha: float) -> Dict:
        try:
            if watermarked_image is None or original_image is None:
                 raise ValueError("Invalid input images")

            # Ensure dimensions match perfectly
            if watermarked_image.shape != original_image.shape:
                watermarked_image = cv2.resize(watermarked_image,
                                              (original_image.shape[1], original_image.shape[0]))

            # Ensure even dimensions for DWT
            h, w = watermarked_image.shape
            h_new, w_new = h - (h % 2), w - (w % 2)
            watermarked_image = watermarked_image[:h_new, :w_new]
            original_image = original_image[:h_new, :w_new]

            # 1. DWT Decomposition
            watermarked_ll = self.dwt.decompose(watermarked_image)['ll_band']
            original_ll = self.dwt.decompose(original_image)['ll_band']
            
            # 2. SVD of Original LL band
            orig_svd = self.svd.decompose(original_ll)
            U_orig = orig_svd['U']
            Vt_orig = orig_svd['V']
            
            if alpha <= 1e-5:
                raise ValueError("Alpha is too close to zero")

            # 3. Inverse Extraction
            # W = U_orig.T @ (LL_marked - LL_orig)/alpha @ Vt_orig.T
            ll_diff = watermarked_ll.astype(np.float64) - original_ll.astype(np.float64)
            extracted_raw = U_orig.T @ (ll_diff / alpha) @ Vt_orig.T
            
            # 4. Post-processing & Denoising
            extracted_watermark = np.abs(extracted_raw)
            
            # Normalize to 0-255 range nicely
            p1, p99 = np.percentile(extracted_watermark, (1, 99))
            if p99 > p1:
                extracted_watermark = 255.0 * (extracted_watermark - p1) / (p99 - p1)
            
            extracted_watermark = np.clip(extracted_watermark, 0, 255).astype(np.uint8)
            
            # --- NEW: Denoising Steps ---
            # 1. Median Blur: Best for removing the "salt-and-pepper" SVD noise
            # Kernel size 3 is usually safe for logos. Use 5 if still very noisy.
            extracted_watermark = cv2.medianBlur(extracted_watermark, 3)
            
            # 2. Optional: Mild Gaussian blur to smooth remaining artifacts
            extracted_watermark = cv2.GaussianBlur(extracted_watermark, (3, 3), 0)
            
            # 3. Final contrast stretch to make logo pop (Otsu's thresholding can also be good here)
            # This snaps blurry grey pixels to black or white
            _, extracted_watermark = cv2.threshold(extracted_watermark, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            return {
                'extracted_watermark': extracted_watermark,
                'extraction_successful': True
            }
            
        except Exception as e:
            self.logger.error(f"Extraction failed: {str(e)}", exc_info=True)
            return {
                'extracted_watermark': None,
                'extraction_successful': False,
                'error': str(e)
            }
