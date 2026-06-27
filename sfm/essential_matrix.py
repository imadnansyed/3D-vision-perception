import numpy as np

class EssentialMatrix:
    @staticmethod
    def compute(F, K1, K2):
        
        E = K2.T @ F @ K1
        
        # rank 2 enforce 
        
        U, S, Vt = np.linalg.svd(E)
        
        S = np.array([1.0, 1.0, 0.0])
        
        E = U @ np.diag(S) @ Vt
        
        return E