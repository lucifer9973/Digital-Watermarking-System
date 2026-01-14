import numpy as np
from typing import Dict, Tuple, Optional

class SVDTransform:
    """
    Singular Value Decomposition for watermark embedding/extraction
    Decomposes matrices into U, S, V components and handles reconstruction
    """
    
    def __init__(self):
        self.U: Optional[np.ndarray] = None
        self.S: Optional[np.ndarray] = None 
        self.V: Optional[np.ndarray] = None
        
    def decompose(self, matrix: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Perform SVD decomposition on input matrix
        """
        try:
            # Use full_matrices=False for proper dimension matching
            U, s, Vt = np.linalg.svd(matrix, full_matrices=False)
            # Convert singular values to diagonal matrix
            S = np.diag(s)
            # Store for later use
            self.U = U
            self.S = s
            self.V = Vt
            return {
                'U': U,
                'S': S,  # Diagonal matrix of singular values
                'V': Vt  # V is already transposed from np.linalg.svd
            }
        except np.linalg.LinAlgError as e:
            raise ValueError(f"SVD decomposition failed: {str(e)}")
            
    def reconstruct(self, U: np.ndarray, S: np.ndarray, V: np.ndarray) -> np.ndarray:
        """
        Reconstruct matrix from U, S, V components
        """
        try:
            U = np.asarray(U)
            S = np.asarray(S)
            V = np.asarray(V)
            
            # V is already transposed from np.linalg.svd
            # Reconstruct: U @ S @ V
            reconstructed = np.dot(U, np.dot(S, V))
            
            if not isinstance(reconstructed, np.ndarray):
                reconstructed = np.asarray(reconstructed)
            
            return reconstructed
        except Exception as e:
            raise ValueError(f"Matrix reconstruction failed: {str(e)}")
            
    def modify_singular_values(self, S: np.ndarray, alpha: float, 
                             watermark: np.ndarray) -> np.ndarray:
        """
        Modify singular values matrix by adding the weighted whole watermark matrix.
        Non-blind approach: S_mod = S + alpha * Watermark
        """
        # Ensure S is a 2D matrix (it should be from decompose())
        if S.ndim == 1:
             S_matrix = np.diag(S)
        else:
             S_matrix = S.copy()
             
        # Ensure watermark matches S dimensions exactly.
        # If it doesn't match, standard broadcasting might fail or give wrong results if not careful.
        # We assume the caller (Embedder) has resized it closely, but we must strictly adhere to S shape.
        # If watermark is larger, crop it. If smaller, pad it.
        # For simplicity, we assume the Embedder did its job, but let's be safe:
        if watermark.shape != S_matrix.shape:
             # This is a critical fallback if embedder didn't resize to S.shape
             # We can't easily cv2.resize here without adding dependency, 
             # so we use basic numpy padding/cropping if strictly necessary.
             rows, cols = S_matrix.shape
             w_rows, w_cols = watermark.shape
             
             new_W = np.zeros((rows, cols), dtype=watermark.dtype)
             min_r, min_c = min(rows, w_rows), min(cols, w_cols)
             new_W[:min_r, :min_c] = watermark[:min_r, :min_c]
             watermark = new_W

        # Direct embedding of the whole watermark matrix
        # S_matrix is diagonal, Watermark is full. modified_S will NOT be strictly diagonal anymore.
        modified_S = S_matrix + (alpha * watermark)
        
        return modified_S
        
    def get_singular_values(self) -> np.ndarray:
        if self.S is None:
            raise ValueError("No SVD decomposition performed yet")
        return np.diag(self.S)