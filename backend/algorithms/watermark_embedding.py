import numpy as np
from typing import Dict, Optional
import logging
import cv2
from .dwt_transform import DWTTransform
from .svd_transform import SVDTransform
from .goa_optimizer import GrasshopperOptimizer
from ..utils.metrics import MetricsCalculator

class WatermarkEmbedder:
    """
    Main watermark embedding class that orchestrates the DWT-SVD-GOA pipeline
    """
    
    def __init__(self, 
                 goa_params: Optional[Dict] = None,
                 dwt_params: Optional[Dict] = None):
        """
        Initialize embedder with optional parameter overrides
        """
        self.dwt = DWTTransform(**(dwt_params or {}))
        self.svd = SVDTransform()
        self.goa = GrasshopperOptimizer(**(goa_params or {}))
        self.metrics = MetricsCalculator()
        self.logger = logging.getLogger(__name__)
        
    def fitness_function(self, alpha) -> float:
        """
        GOA fitness function based on PSNR and SSIM
        """
        if not hasattr(self, '_temp_data'):
            raise RuntimeError("Temporary embedding data not set")
        
        # Helper function to safely get shape
        def safe_shape_fitness(obj, name="object"):
            if isinstance(obj, tuple):
                raise ValueError(f"{name} is a tuple when numpy array expected in fitness function")
            if not hasattr(obj, 'shape'):
                raise ValueError(f"{name} does not have 'shape' attribute in fitness function. Type: {type(obj)}")
            return obj.shape
        
        # Ensure alpha is a scalar float
        if isinstance(alpha, (list, np.ndarray)):
            alpha = float(alpha[0] if len(alpha) > 0 else 0.1)
        else:
            alpha = float(alpha)
        
        # Ensure alpha is in valid range
        alpha = max(0.01, min(1.0, alpha))
            
        try:
            # Perform trial embedding with current alpha
            modified_S = self.svd.modify_singular_values(
                self._temp_data['S'],
                alpha,
                self._temp_data['resized_watermark']
            )
            
            # Reconstruct LL band
            modified_ll = self.svd.reconstruct(
                self._temp_data['U'],
                modified_S,
                self._temp_data['V']
            )
            
            # Ensure modified_ll is a numpy array
            if not isinstance(modified_ll, np.ndarray):
                raise ValueError(f"Expected numpy array from SVD reconstruction, got {type(modified_ll)}")
            
            # Ensure ll_band in temp_data is a numpy array (not a tuple)
            ll_band_ref = self._temp_data['ll_band']
            if not isinstance(ll_band_ref, np.ndarray):
                if isinstance(ll_band_ref, tuple):
                    raise ValueError(f"ll_band is a tuple instead of numpy array. This should not happen. Type: {type(ll_band_ref)}")
                ll_band_ref = np.asarray(ll_band_ref)
                self._temp_data['ll_band'] = ll_band_ref
            
            # Use safe_shape to get shapes
            ll_band_shape_fitness = safe_shape_fitness(ll_band_ref, "ll_band (fitness)")
            modified_ll_shape_fitness = safe_shape_fitness(modified_ll, "modified_ll (fitness)")
            
            # Ensure modified_ll matches original LL band shape
            if modified_ll_shape_fitness != ll_band_shape_fitness:
                modified_ll = cv2.resize(modified_ll,
                                       (ll_band_shape_fitness[1],
                                        ll_band_shape_fitness[0]))
            
            # Double-check it's still a numpy array after resize
            if not isinstance(modified_ll, np.ndarray):
                raise ValueError(f"Expected numpy array after resize, got {type(modified_ll)}")
            
            # Update LL band in DWT coefficients (deep copy to avoid modifying original)
            trial_coeffs = []
            for i, coeff in enumerate(self._temp_data['dwt_coeffs']):
                if i == 0:
                    trial_coeffs.append(modified_ll.copy())
                elif isinstance(coeff, tuple):
                    # For detail bands, copy the tuple
                    trial_coeffs.append(tuple(c.copy() if hasattr(c, 'copy') else c for c in coeff))
                else:
                    trial_coeffs.append(coeff.copy() if hasattr(coeff, 'copy') else coeff)
            
            # Reconstruct full image
            reconstructed = self.dwt.reconstruct(trial_coeffs)
            
            # Ensure reconstructed is a numpy array
            if not isinstance(reconstructed, np.ndarray):
                raise ValueError(f"Expected numpy array from DWT reconstruction, got {type(reconstructed)}")
            
            # Ensure original in temp_data is a numpy array
            original_ref = self._temp_data['original']
            if not isinstance(original_ref, np.ndarray):
                if isinstance(original_ref, tuple):
                    raise ValueError(f"original is a tuple instead of numpy array. Type: {type(original_ref)}")
                original_ref = np.asarray(original_ref)
                self._temp_data['original'] = original_ref
            
            # Use safe_shape to get shapes
            reconstructed_shape = safe_shape_fitness(reconstructed, "reconstructed (fitness)")
            original_shape = safe_shape_fitness(original_ref, "original (fitness)")
            
            # Ensure dimensions match and values are valid
            if reconstructed_shape != original_shape:
                reconstructed = cv2.resize(reconstructed, 
                                         (original_shape[1], 
                                          original_shape[0]),
                                         interpolation=cv2.INTER_CUBIC)
            
            # Ensure it's still a numpy array after resize
            if not isinstance(reconstructed, np.ndarray):
                raise ValueError(f"Expected numpy array after resize, got {type(reconstructed)}")
            
            # Clip to valid range
            reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
            original = self._temp_data['original'].astype(np.uint8)
            
            # Calculate quality metrics
            metrics = self.metrics.calculate_all_metrics(original, reconstructed)
            
            # Weighted fitness (lower is better, negative because we maximize PSNR and SSIM)
            fitness = -(0.7 * metrics['psnr'] + 0.3 * metrics['ssim'])
            
            # Return finite value
            return fitness if np.isfinite(fitness) else 1e10
            
        except Exception as e:
            self.logger.error(f"Fitness function error: {str(e)}", exc_info=True)
            # Return a very bad fitness value if something goes wrong
            return 1e10
        
    def embed(self, cover_image: np.ndarray, 
             watermark: np.ndarray) -> Dict:
        """
        Embed watermark into cover image using DWT-SVD-GOA
        
        Args:
            cover_image: Original image to watermark
            watermark: Watermark image to embed
            
        Returns:
            Dict containing:
                - watermarked_image: Final watermarked image
                - alpha: Optimal scaling factor
                - metrics: Quality metrics
        """
        try:
            # Helper function to safely get shape with better error messages
            def safe_shape(obj, name="object"):
                if isinstance(obj, tuple):
                    raise ValueError(f"{name} is a tuple when numpy array expected. Tuple contents: {[type(x) for x in obj]}")
                if not hasattr(obj, 'shape'):
                    raise ValueError(f"{name} does not have 'shape' attribute. Type: {type(obj)}")
                return obj.shape
            # Ensure images are valid
            if cover_image is None or cover_image.size == 0:
                raise ValueError("Cover image is invalid")
            if watermark is None or watermark.size == 0:
                raise ValueError("Watermark image is invalid")
                
            # Ensure image dimensions are even for DWT
            cover_shape = safe_shape(cover_image, "cover_image")
            if cover_shape[0] % 2 != 0 or cover_shape[1] % 2 != 0:
                # Resize to even dimensions
                cover_image = cv2.resize(cover_image, 
                                       (cover_shape[1] // 2 * 2, 
                                        cover_shape[0] // 2 * 2))
            
            # 1. DWT Decomposition
            dwt_components = self.dwt.decompose(cover_image)
            ll_band = dwt_components['ll_band']
            
            # Ensure ll_band is a numpy array, not a tuple
            if isinstance(ll_band, tuple):
                self.logger.error(f"ll_band is a tuple! This should not happen. DWT decompose returned: {type(ll_band)}")
                raise ValueError(f"LL band from DWT decomposition is a tuple instead of numpy array. Got: {type(ll_band)}, value type: {type(ll_band[0]) if len(ll_band) > 0 else 'empty'}")
            
            if not isinstance(ll_band, np.ndarray):
                self.logger.warning(f"ll_band is not a numpy array, converting. Type: {type(ll_band)}")
                ll_band = np.asarray(ll_band)
            
            if ll_band.ndim != 2:
                raise ValueError(f"Expected 2D array for LL band, got {ll_band.ndim}D array with shape {ll_band.shape}")
            
            # Verify ll_band has shape attribute before using it
            if not hasattr(ll_band, 'shape'):
                raise ValueError(f"ll_band does not have 'shape' attribute. Type: {type(ll_band)}")
            
            # Resize watermark to match LL band dimensions
            if not isinstance(watermark, np.ndarray):
                watermark = np.asarray(watermark)
            if watermark.ndim != 2:
                # If watermark is not 2D, convert it
                if watermark.ndim == 3:
                    watermark = cv2.cvtColor(watermark, cv2.COLOR_BGR2GRAY)
                else:
                    raise ValueError(f"Watermark must be 2D or 3D array, got {watermark.ndim}D")
            
            # Double-check ll_band is still an array before accessing shape
            if not isinstance(ll_band, np.ndarray):
                raise ValueError(f"ll_band became non-array after validation. Type: {type(ll_band)}")
            
            # Use safe_shape to get ll_band dimensions
            ll_band_shape = safe_shape(ll_band, "ll_band")
            resized_watermark = cv2.resize(watermark, 
                                         (ll_band_shape[1], ll_band_shape[0]),
                                         interpolation=cv2.INTER_AREA)
            
            # Ensure resized_watermark is a numpy array
            if not isinstance(resized_watermark, np.ndarray):
                resized_watermark = np.asarray(resized_watermark)
            
            # 2. SVD on LL band
            svd_components = self.svd.decompose(ll_band)
            
            # Ensure all temp data components are numpy arrays (not tuples)
            # Verify dwt_coeffs structure
            if not isinstance(dwt_components['coefficients'], list):
                raise ValueError("DWT coefficients must be a list")
            if len(dwt_components['coefficients']) == 0:
                raise ValueError("DWT coefficients list is empty")
            if not isinstance(dwt_components['coefficients'][0], np.ndarray):
                raise ValueError(f"First DWT coefficient must be numpy array, got {type(dwt_components['coefficients'][0])}")
            
            # Store data for fitness function
            self._temp_data = {
                'U': np.asarray(svd_components['U']),
                'S': np.asarray(svd_components['S']),
                'V': np.asarray(svd_components['V']),
                'watermark': np.asarray(watermark),
                'resized_watermark': np.asarray(resized_watermark),
                'original': np.asarray(cover_image),
                'dwt_coeffs': dwt_components['coefficients'],  # Keep as list with first element array, rest tuples
                'll_band': np.asarray(ll_band)  # Ensure it's an array
            }
            
            # Verify ll_band is still an array after conversion
            if not isinstance(self._temp_data['ll_band'], np.ndarray):
                raise ValueError(f"LL band must be numpy array, got {type(self._temp_data['ll_band'])}")
            
            # 3. Find optimal alpha using GOA (use smaller iterations for faster processing)
            # Reduce GOA iterations for web app responsiveness
            original_max_iter = self.goa.max_iterations
            self.goa.max_iterations = min(20, original_max_iter)  # Limit to 20 iterations for web app
            
            try:
                goa_result = self.goa.optimize(self.fitness_function)
                optimal_alpha = float(goa_result['best_solution'][0])
            except Exception as e:
                self.logger.warning(f"GOA optimization failed: {str(e)}, using default alpha=0.1")
                optimal_alpha = 0.1
            finally:
                self.goa.max_iterations = original_max_iter
            
            # Ensure alpha is in valid range
            optimal_alpha = max(0.01, min(1.0, optimal_alpha))
            
            # 4. Final embedding with optimal alpha
            modified_S = self.svd.modify_singular_values(
                svd_components['S'],
                optimal_alpha,
                resized_watermark
            )
            
            # 5. Reconstruct watermarked LL band
            modified_ll = self.svd.reconstruct(
                svd_components['U'],
                modified_S,
                svd_components['V']
            )
            
            # Ensure modified_ll is a numpy array
            if not isinstance(modified_ll, np.ndarray):
                raise ValueError(f"Expected numpy array from SVD reconstruction, got {type(modified_ll)}")
            
            # Ensure modified_ll has correct shape
            ll_band_shape_final = safe_shape(ll_band, "ll_band (final check)")
            modified_ll_shape = safe_shape(modified_ll, "modified_ll")
            if modified_ll_shape != ll_band_shape_final:
                modified_ll = cv2.resize(modified_ll, 
                                       (ll_band_shape_final[1], ll_band_shape_final[0]))
            
            # Ensure it's still a numpy array after resize
            if not isinstance(modified_ll, np.ndarray):
                raise ValueError(f"Expected numpy array after resize, got {type(modified_ll)}")
            
            # Update LL band in DWT coefficients (must be numpy array, not tuple)
            if not isinstance(dwt_components['coefficients'], list):
                raise ValueError("DWT coefficients must be a list")
            if len(dwt_components['coefficients']) == 0:
                raise ValueError("DWT coefficients list is empty")
            
            # Ensure first element is replaced with numpy array
            dwt_components['coefficients'][0] = modified_ll
            
            # 6. Inverse DWT
            watermarked_image = self.dwt.reconstruct(dwt_components['coefficients'])
            
            # Ensure watermarked image has correct shape and type
            watermarked_shape = safe_shape(watermarked_image, "watermarked_image")
            cover_shape_final = safe_shape(cover_image, "cover_image (final)")
            if watermarked_shape != cover_shape_final:
                watermarked_image = cv2.resize(watermarked_image,
                                             (cover_shape_final[1], cover_shape_final[0]))
            
            # Clip values to valid range [0, 255]
            watermarked_image = np.clip(watermarked_image, 0, 255).astype(np.uint8)
            
            # Calculate final metrics
            final_metrics = self.metrics.calculate_all_metrics(
                cover_image, 
                watermarked_image
            )
            
            return {
                'watermarked_image': watermarked_image,
                'alpha': optimal_alpha,
                'metrics': final_metrics
            }
            
        except Exception as e:
            self.logger.error(f"Embedding failed: {str(e)}", exc_info=True)
            raise
        finally:
            # Cleanup temporary data
            if hasattr(self, '_temp_data'):
                del self._temp_data