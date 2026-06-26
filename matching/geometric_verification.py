import cv2
import numpy as np

from matching.match_set import MatchSet


class GeometricVerification:
    
    def __init__(self, ransac_threshold = 1.0, confidence = 0.99):
        
        self.ransac_threshold = ransac_threshold
        self.confidence = confidence
        
        
    def verify(self, feature1, feature2, good_matches):
        
        pts1 = np.float32([
            feature1.keypoints[m.queryIdx].pt for m in good_matches
        ])
        
        pts2 = np.float32([
            feature2.keypoints[m.trainIdx].pt for m in good_matches
        ])
        
        F, mask = cv2.findFundamentalMat(
            pts1, pts2, cv2.FM_RANSAC, self.ransac_threshold, self.confidence
        )
        
        mask = mask.ravel().astype(bool)
        
        
        inlier_matches = [
            m for m, keep in zip(good_matches, mask)
            if keep
        ]
        
        pts1 = pts1[mask]
        pts2 = pts2[mask]
        
        return MatchSet(
            image1_id=feature1.image_id,
            image2_id=feature2.image_id,
            matches=inlier_matches,
            pts1=pts1,
            pts2=pts2,
            F=F,
            inlier_mask=mask
        )