class IncrementalTriangulation:

    def triangulate(
        self,
        reconstruction,
        registered_image_id,
        new_image_id,
        match_set,
        P1,
        P2
    ):

        import cv2
import cv2
import numpy as np

from sfm.track import Track


class IncrementalTriangulation:

    def triangulate(
        self,
        reconstruction,
        match_set,
        feature_database,
        K1,
        K2,
        R1,
        t1,
        R2,
        t2
    ):

        # Projection matrices
        P1 = K1 @ np.hstack((R1, t1.reshape(3, 1)))
        P2 = K2 @ np.hstack((R2, t2.reshape(3, 1)))

        image1_id = match_set.image1_id
        image2_id = match_set.image2_id

        for match in match_set.matches:

            kp1_idx = match.queryIdx
            kp2_idx = match.trainIdx

            # --------------------------------------------------
            # Case 1 : Already reconstructed
            # --------------------------------------------------

            if (image1_id, kp1_idx) in reconstruction.feature_to_point:

                point_id = reconstruction.feature_to_point[
                    (image1_id, kp1_idx)
                ]

                track = reconstruction.tracks[point_id]

                if image2_id not in track.observation:

                    track.add_observation(
                        image2_id,
                        kp2_idx
                    )

                    reconstruction.feature_to_point[
                        (image2_id, kp2_idx)
                    ] = point_id

                continue

            # --------------------------------------------------
            # Case 2 : New point
            # --------------------------------------------------

            pt1 = np.array(
                feature_database[image1_id]
                .keypoints[kp1_idx]
                .pt,
                dtype=np.float32
            ).reshape(2, 1)

            pt2 = np.array(
                feature_database[image2_id]
                .keypoints[kp2_idx]
                .pt,
                dtype=np.float32
            ).reshape(2, 1)

            X = cv2.triangulatePoints(
                P1,
                P2,
                pt1,
                pt2
            )

            X = X[:3] / X[3]
            X = X.flatten()

            point_id = len(reconstruction.points3D)

            reconstruction.points3D = np.vstack([
    reconstruction.points3D,
    X
])

            track = Track(point_id)

            track.add_observation(
                image1_id,
                kp1_idx
            )

            track.add_observation(
                image2_id,
                kp2_idx
            )

            reconstruction.tracks.append(track)

            reconstruction.feature_to_point[
                (image1_id, kp1_idx)
            ] = point_id

            reconstruction.feature_to_point[
                (image2_id, kp2_idx)
            ] = point_id

        return reconstruction