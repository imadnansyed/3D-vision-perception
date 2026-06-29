from sfm.track import Track


class TrackBuilder:
    @staticmethod
    def build(initial_pair, points3D):
        
        tracks = []
        
        for point_id, match in enumerate(initial_pair.matches):
            track = Track(point_id=point_id)
            
            # add obeservation of image1
            track.add_observation(initial_pair.image1_id, match.queryIdx)
            
            # add obeservation of image2
            track.add_observation(initial_pair.image2_id, match.trainIdx)
            
            tracks.append(track)
        
        return tracks
        