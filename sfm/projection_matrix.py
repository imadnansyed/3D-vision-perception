import numpy as np

class ProjectionMatrix:
    @staticmethod
    def compute_projection_matrix(K, R, t:np.ndarray):
        Rt = np.hstack((R, t.reshape(3,1)))
        
        P = K @ Rt
        
        return P