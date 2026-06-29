import numpy as np
from sfm.pose_recovery import CheiralityCheck


class Triangulation:
    def __init__(self):
        self.triangulator = CheiralityCheck()
    
    def triangulate(self, P1, P2, pts1, pts2):
        
        points3D = []
        
        for pt1, pt2 in zip(pts1, pts2):
            point3d = self.triangulator.triangulate_points(P1, P2, pt1, pt2)
            
            points3D.append(point3d)
            
        return np.asarray(points3D)