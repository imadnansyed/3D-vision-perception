from dataclasses import dataclass
import numpy as np


@dataclass
class MatchSet:
    
    image1_id : str
    image2_id : str
    
    matches : list
    
    pts1 : np.ndarray
    pts2 : np.ndarray
    
    F : np.ndarray
    
    inlier_mask : np.ndarray
    