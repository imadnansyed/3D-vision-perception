class Track:
    def __init__(self, point_id):
        self.point_id = point_id
        self.observation = {}
    
    def add_observation(self, image_id, keypoint_idx):
        self.observation[image_id] = keypoint_idx