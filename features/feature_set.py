from dataclasses import dataclass
import numpy as np

@dataclass #auto creates constructor, __repr__, __eq__, etc.
class FeatureSet:
    image_id : str
    image_name : str
    keypoints : list
    descriptors : np.ndarray
    
