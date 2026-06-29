import cv2


class PnPSolver:
    
    @staticmethod
    def solve(points3D, image_points, K):
        
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints=points3D,
            imagePoints=image_points,
            cameraMatrix=K,
            distCoeffs=None,
            reprojectionError=4.0,
            confidence=0.99,
            flags=cv2.SOLVEPNP_EPNP
        )
        
        if not success:
            return None 
        
        R, _ = cv2.Rodrigues(rvec)
        
        return {
            "R": R,
            "t": tvec.reshape(3),
            "inliers": inliers
        }