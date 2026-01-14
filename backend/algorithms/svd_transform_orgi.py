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
        
        Args:
            matrix (np.ndarray): Input matrix for SVD
            
        Returns:
            Dict containing:
                - U: Left singular vectors
                - S: Singular values (diagonal matrix)
                - V: Right singular vectors (transposed)
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
        
        Args:
            U (np.ndarray): Left singular vectors
            S (np.ndarray): Singular values (diagonal matrix)
            V (np.ndarray): Right singular vectors (transposed)
            
        Returns:
            np.ndarray: Reconstructed matrix
        """
        try:
            # Ensure inputs are numpy arrays
            U = np.asarray(U)
            S = np.asarray(S)
            V = np.asarray(V)
            
            # V is already transposed from np.linalg.svd
            # Reconstruct: U @ S @ V (where V is Vt)
            reconstructed = np.dot(U, np.dot(S, V))
            
            # Ensure we got a numpy array
            if not isinstance(reconstructed, np.ndarray):
                reconstructed = np.asarray(reconstructed)
            
            return reconstructed
        except Exception as e:
            raise ValueError(f"Matrix reconstruction failed: {str(e)}")
            
    def modify_singular_values(self, S: np.ndarray, alpha: float, 
                             watermark: np.ndarray) -> np.ndarray:
        """
        Modify singular values for watermark embedding.
        Standard DWT-SVD approach: Modify S by adding scaled watermark.
        
        Args:
            S (np.ndarray): Original singular values matrix (diagonal, 2D)
            alpha (float): Scaling factor from GOA
            watermark (np.ndarray): Watermark to embed (must match LL band shape)
            
        Returns:
            np.ndarray: Modified singular values matrix with same shape as S
        """
        # Extract diagonal elements from S (singular values)
        # CRITICAL: Make a writable copy, not a view
        if S.ndim == 2:
            s_values = np.diag(S).copy()  # .copy() ensures writable array
            s_shape = S.shape
        else:
            s_values = S.copy()  # Already a copy, but ensure it's writable
            s_shape = (len(s_values), len(s_values))
        
        # Ensure the array is writable
        if not s_values.flags.writeable:
            s_values = s_values.copy()
        
        # Perform SVD on watermark to get its singular values
        try:
            U_w, s_w, Vt_w = np.linalg.svd(watermark, full_matrices=False)
            # Ensure s_w is writable if we need to modify it later
            if not s_w.flags.writeable:
                s_w = s_w.copy()
        except Exception:
            # If SVD fails, use watermark directly (flattened)
            s_w = watermark.flatten()
            if not isinstance(s_w, np.ndarray):
                s_w = np.asarray(s_w)
        
        # Modify singular values: add scaled watermark singular values
        # Take minimum length to avoid index errors
        min_len = min(len(s_values), len(s_w))
        if min_len > 0:
            # Use in-place addition with explicit copy to ensure writability
            s_values = s_values.copy()  # Ensure writable
            s_values[:min_len] = s_values[:min_len] + alpha * s_w[:min_len]
        
        # Ensure singular values are non-negative
        s_values = np.maximum(s_values, 0)
        
        # Reconstruct modified S matrix with original dimensions
        modified_S = np.diag(s_values)
        
        # Ensure dimensions match original S
        if modified_S.shape != s_shape:
            # Create new matrix with correct shape
            new_S = np.zeros(s_shape, dtype=modified_S.dtype)
            min_rows = min(modified_S.shape[0], s_shape[0])
            min_cols = min(modified_S.shape[1], s_shape[1])
            new_S[:min_rows, :min_cols] = modified_S[:min_rows, :min_cols]
            modified_S = new_S
            
        return modified_S
        
    def get_singular_values(self) -> np.ndarray:
        """
        Get current singular values
        
        Returns:
            np.ndarray: Diagonal matrix of singular values
        """
        if self.S is None:
            raise ValueError("No SVD decomposition performed yet")
        return np.diag(self.S)