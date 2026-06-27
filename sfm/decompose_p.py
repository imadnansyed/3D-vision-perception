import numpy as np
from scipy.linalg import rq
from scipy.io import loadmat


class DecomposeP:
    
    @staticmethod
    def decompose_projection_matrix(P):
        
        M = P[:, :3]

        K, R = rq(M)

        T = np.diag(np.where(np.diag(K) >= 0, 1, -1))
        
        K = K @ T
        R = T @ R

        K = K / K[2, 2]

        C = -np.linalg.inv(M) @ P[:, 3]


        t = -R @ C

        return K, R, t, C

    @staticmethod
    def get_camera_matrices():
        data = loadmat('sfm\dino.mat')    
        P = data["P"]

        camera_matrices = []

        for i in range(P.shape[1]):
            Pi = P[0, i]      # 3x4 projection matrix

            # Decompose
            K, R, t, C = DecomposeP.decompose_projection_matrix(Pi)

            camera = {
                "id": i,
                "P": None,
                "K": K,
                "R": None,
                "t": None,
                "C": None
            }
            
            camera_matrices.append(camera)
            
        return camera_matrices