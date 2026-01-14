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
    Uses reverse embedding formula: W = U_orig.T @ (LL_marked - LL_orig)/alpha @ V_orig.T
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
        """
        Extract watermark from watermarked image
        """
        try:
            # Ensure images are valid
            if watermarked_image is None or original_image is None:
                 raise ValueError("Invalid input images")

            # Ensure dimensions match perfectly
            if watermarked_image.shape != original_image.shape:
                watermarked_image = cv2.resize(watermarked_image,
                                              (original_image.shape[1], original_image.shape[0]))

            # Ensure even dimensions for DWT
            h, w = watermarked_image.shape
            h_new, w_new = h - (h % 2), w - (w % 2)
            if h_new != h or w_new != w:
                 watermarked_image = watermarked_image[:h_new, :w_new]
                 original_image = original_image[:h_new, :w_new]

            # 1. DWT Decomposition
            watermarked_ll = self.dwt.decompose(watermarked_image)['ll_band']
            original_ll = self.dwt.decompose(original_image)['ll_band']
            
            # 2. SVD of Original LL band ONLY
            # We need U and V from the original image to reverse the embedding
            orig_svd = self.svd.decompose(original_ll)
            U_orig = orig_svd['U']
            Vt_orig = orig_svd['V'] # This is V transpose
            
            if alpha <= 1e-5:
                raise ValueError("Alpha is too close to zero")

            # 3. Extraction using the inverse formula
            # W = U_orig.T @ (LL_marked - LL_orig)/alpha @ Vt_orig.T
            
            # Calculate difference in LL bands
            ll_diff = watermarked_ll.astype(np.float64) - original_ll.astype(np.float64)
            
            # Scale by alpha
            W_scaled = ll_diff / alpha
            
            # Project back to watermark space
            extracted_watermark = U_orig.T @ W_scaled @ Vt_orig.T
            
            # 4. ROBUST POST-PROCESSING
            # Use absolute values as SVD sign ambiguity can flip colours
            extracted_watermark = np.abs(extracted_watermark)
            
            # Robust normalization using percentiles to ignore outliers
            # This fixes the "totally dark" issue by stretching the contrast 
            # of the main signal, ignoring extreme noise pixels.
            p5, p95 = np.percentile(extracted_watermark, (5, 95))
            
            if p95 > p5:
                # Stretch the middle 90% of data to cover full 0-255 range
                extracted_watermark = 255.0 * (extracted_watermark - p5) / (p95 - p5)
            else:
                # Fallback to min/max if percentiles are too close
                p_min, p_max = extracted_watermark.min(), extracted_watermark.max()
                if p_max > p_min:
                     extracted_watermark = 255.0 * (extracted_watermark - p_min) / (p_max - p_min)

            # Clip values that fell outside the percentiles
            extracted_watermark = np.clip(extracted_watermark, 0, 255).astype(np.uint8)
            
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