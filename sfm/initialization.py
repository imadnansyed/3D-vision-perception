import numpy as np

from sfm.essential_matrix import EssentialMatrix
from sfm.initial_pair_selection import InitialPairSelection
from sfm.pose_recovery import CheiralityCheck
from sfm.projection_matrix import ProjectionMatrix
from sfm.reconstruction import Reconstruction
from sfm.track_builder import TrackBuilder
from sfm.triangulation import Triangulation


class Initializor:
    
    def __init__(self):
        self.initial_pair_selector = InitialPairSelection()
        self.pose_recovery = CheiralityCheck()
        self.triangulator = Triangulation()
        self.reconstruction = Reconstruction()
        self.track_builder = TrackBuilder()
    
    def initialize(self, feature_database, camera_database, match_database):

        # ---------------------------------------
        # Select initial pair
        # ---------------------------------------

        initial_pair = self.initial_pair_selector.select(
            match_database
        )

        print(
            f"Initial Pair : "
            f"{initial_pair.image1_id} <-> {initial_pair.image2_id}"
        )

        K1 = camera_database[
            initial_pair.image1_id
        ]["K"]

        K2 = camera_database[
            initial_pair.image2_id
        ]["K"]

        # ---------------------------------------
        # Essential Matrix
        # ---------------------------------------

        E = EssentialMatrix.compute(
            initial_pair.F,
            K1,
            K2
        )

        # ---------------------------------------
        # Recover Pose
        # ---------------------------------------

        R, t = self.pose_recovery.check(
            initial_pair.pts1,
            initial_pair.pts2,
            K1,
            K2,
            E
        )

        # ---------------------------------------
        # Projection Matrices
        # ---------------------------------------

        P1 = ProjectionMatrix.compute_projection_matrix(
            K1,
            np.eye(3),
            np.zeros(3)
        )

        P2 = ProjectionMatrix.compute_projection_matrix(
            K2,
            R,
            t
        )

        # ---------------------------------------
        # Triangulation
        # ---------------------------------------

        points3D = self.triangulator.triangulate(
            P1,
            P2,
            initial_pair.pts1,
            initial_pair.pts2
        )

        # ---------------------------------------
        # Reconstruction
        # ---------------------------------------

        self.reconstruction.add_camera(
            np.eye(3),
            np.zeros(3),
            initial_pair.image1_id
        )

        self.reconstruction.add_camera(
            R,
            t,
            initial_pair.image2_id
        )

        self.reconstruction.set_points(points3D)

        tracks = self.track_builder.build(
            initial_pair,
            points3D
        )

        self.reconstruction.add_tracks(tracks)

        print(f"Initial Points : {len(points3D)}")

        return self.reconstruction