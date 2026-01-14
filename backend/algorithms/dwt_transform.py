import numpy as np
import pywt
from typing import Dict, Tuple, List, Optional

class DWTTransform:
    """
    Discrete Wavelet Transform for 2-level Haar decomposition
    Used for multi-resolution analysis in watermarking
    """
    
    def __init__(self, wavelet: str = 'haar', level: int = 2):
        """
        Initialize DWT transformer
        
        Args:
            wavelet (str): Wavelet type to use (default: 'haar')
            level (int): Number of decomposition levels (default: 2)
        """
        self.wavelet = wavelet
        self.level = level
        self.decomposition_structure = None
    
    def decompose(self, image: np.ndarray) -> Dict:
        """
        Perform 2-level DWT decomposition on input image
        
        Args:
            image (np.ndarray): Input image array (grayscale)
            
        Returns:
            Dict containing:
                - coefficients: All wavelet coefficients
                - structure: Decomposition structure info
                - ll_band: LL2 sub-band for watermark embedding
        
        Raises:
            ValueError: If image dimensions are not even
        """
        if image.shape[0] % 2 != 0 or image.shape[1] % 2 != 0:
            raise ValueError("Image dimensions must be even")
            
        # Perform 2-level decomposition
        coeffs = pywt.wavedec2(image, self.wavelet, level=self.level)
        
        # Verify coeffs structure
        if not isinstance(coeffs, (list, tuple)):
            raise ValueError(f"DWT decomposition returned unexpected type: {type(coeffs)}")
        if len(coeffs) == 0:
            raise ValueError("DWT decomposition returned empty coefficients")
        
        # Verify first element is a numpy array (LL band)
        if isinstance(coeffs[0], tuple):
            raise ValueError(f"First DWT coefficient (LL band) is a tuple! This should be a numpy array. Got: {type(coeffs[0])}")
        if not isinstance(coeffs[0], np.ndarray):
            raise ValueError(f"First DWT coefficient (LL band) must be numpy array, got {type(coeffs[0])}")
        if not hasattr(coeffs[0], 'shape'):
            raise ValueError(f"First DWT coefficient (LL band) does not have 'shape' attribute. Type: {type(coeffs[0])}")
        
        # Store decomposition structure
        # First element is LL band (array), rest are tuples of (LH, HL, HH)
        self.decomposition_structure = []
        for i, c in enumerate(coeffs):
            if i == 0:
                # LL band is a numpy array - verify before accessing shape
                if not isinstance(c, np.ndarray):
                    raise ValueError(f"LL band at index 0 is not a numpy array: {type(c)}")
                if not hasattr(c, 'shape'):
                    raise ValueError(f"LL band does not have 'shape' attribute: {type(c)}")
                self.decomposition_structure.append(c.shape)
            else:
                # Detail bands are tuples of (LH, HL, HH) arrays
                if isinstance(c, tuple):
                    self.decomposition_structure.append(tuple(arr.shape for arr in c if hasattr(arr, 'shape')))
                else:
                    self.decomposition_structure.append(c.shape if hasattr(c, 'shape') else None)
        
        # Extract LL2 band - ensure it's an array
        ll2_band = coeffs[0]
        if not isinstance(ll2_band, np.ndarray):
            raise ValueError(f"LL band extraction failed: got {type(ll2_band)} instead of numpy array")
        if not hasattr(ll2_band, 'shape'):
            raise ValueError(f"LL band does not have 'shape' attribute: {type(ll2_band)}")
        
        return {
            'coefficients': coeffs,
            'structure': self.decomposition_structure,
            'll_band': ll2_band
        }
    
    def reconstruct(self, coefficients: List) -> np.ndarray:
        """
        Reconstruct image from DWT coefficients
        
        Args:
            coefficients (List): DWT coefficients from decompose()
            
        Returns:
            np.ndarray: Reconstructed image
            
        Raises:
            ValueError: If decomposition structure is missing
        """
        if self.decomposition_structure is None:
            raise ValueError("No decomposition structure found")
        
        # Ensure coefficients are in the correct format
        # First element should be a numpy array (LL band)
        # Subsequent elements should be tuples of (LH, HL, HH) bands
        if not isinstance(coefficients, list):
            raise ValueError("Coefficients must be a list")
        
        if len(coefficients) == 0:
            raise ValueError("Coefficients list is empty")
        
        # Ensure first element is a numpy array
        if not isinstance(coefficients[0], np.ndarray):
            raise ValueError(f"First coefficient must be a numpy array, got {type(coefficients[0])}")
        
        # Reconstruct using PyWavelets
        reconstructed = pywt.waverec2(coefficients, self.wavelet)
        
        # Ensure we got a numpy array back
        if not isinstance(reconstructed, np.ndarray):
            raise ValueError(f"Reconstruction returned {type(reconstructed)}, expected numpy array")
        
        return reconstructed
    
    def get_ll_band(self, coefficients: List) -> np.ndarray:
        """
        Extract LL2 sub-band from coefficients
        
        Args:
            coefficients (List): DWT coefficients
            
        Returns:
            np.ndarray: LL2 sub-band coefficients
        """
        return coefficients[0]