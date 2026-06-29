import numpy as np


class CorrespondenceFinder:

    @staticmethod
    def find(
        reconstruction,
        next_image_id,
        match_database,
        feature_database
    ):

        object_points = []
        image_points = []

        for track in reconstruction.tracks:

            point3D = reconstruction.points3D[
                track.point_id
            ]

            found = False

            for registered_image, kp_idx in track.observation.items():

                if found:
                    break

                for match_set in match_database:

                    # ----------------------------------
                    # registered ---> next
                    # ----------------------------------

                    if (
                        match_set.image1_id == registered_image
                        and
                        match_set.image2_id == next_image_id
                    ):

                        for m in match_set.matches:

                            if m.queryIdx == kp_idx:

                                pixel = feature_database[
                                    next_image_id
                                ].keypoints[
                                    m.trainIdx
                                ].pt

                                object_points.append(point3D)
                                image_points.append(pixel)

                                found = True
                                break

                    # ----------------------------------
                    # next ---> registered
                    # ----------------------------------

                    elif (
                        match_set.image2_id == registered_image
                        and
                        match_set.image1_id == next_image_id
                    ):

                        for m in match_set.matches:

                            if m.trainIdx == kp_idx:

                                pixel = feature_database[
                                    next_image_id
                                ].keypoints[
                                    m.queryIdx
                                ].pt

                                object_points.append(point3D)
                                image_points.append(pixel)

                                found = True
                                break

        return (
            np.asarray(object_points, dtype=np.float32),
            np.asarray(image_points, dtype=np.float32)
        )