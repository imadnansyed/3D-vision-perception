import cv2


class Matcher:
        
    def match(self, descriptors1, descriptors2):
    
        FLANN_INDEX_KDTREE = 1

        index_params = dict(
            algorithm=FLANN_INDEX_KDTREE,
            trees=5
        )

        search_params = dict(checks=50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        matches = flann.knnMatch(descriptors1, descriptors2, k=2)
        
        return matches