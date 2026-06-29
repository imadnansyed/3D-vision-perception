class Reconstruction:
    
    def __init__(self):
        self.cameras = {}
        self.points3D = None
        self.tracks = []
        
        # (image_id, keypoint_idx) -> point_id
        self.feature_to_point = {}
        
    def add_camera(self, R, t, image_id):
        self.cameras[image_id] = {
            "R": R,
            "t": t
        }
    
    def set_points(self, points3D):
        self.points3D = points3D
        
    def add_tracks(self, tracks):
        for track in tracks:
            self.tracks.append(track)

            for image_id, kp_idx in track.observation.items():

                self.feature_to_point[(image_id, kp_idx)] = track.point_id
        
        