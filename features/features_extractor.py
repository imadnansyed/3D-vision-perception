import cv2
import numpy as np
from features.feature_set import FeatureSet

class FeaturesExtractor:
    def __init__(self):
        self.detector = cv2.SIFT_create()
        
    def extract(self, images):
        
        feature_database = []
        
        for image in images:
            keypoints, descriptors = self.detector.detectAndCompute(image["gray"], None)
            feature_set = FeatureSet(
                image_id = image["id"],
                image_name = image["name"],
                keypoints = keypoints,
                descriptors = descriptors
            )
            feature_database.append(feature_set)
            
        return feature_database
