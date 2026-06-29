import numpy as np

from sfm.initialization import Initializor
from sfm.reconstruction import Reconstruction

from sfm.next_image_selector import NextImageSelector
from sfm.correspondence_finder import CorrespondenceFinder
from sfm.pnp_solver import PnPSolver
from sfm.incremental_triangulatin import IncrementalTriangulation


class IncrementalSfM:

    def __init__(
        self,
        feature_database,
        camera_database,
        match_database
    ):

        # Databases
        self.feature_database = feature_database
        self.camera_database = camera_database
        self.match_database = match_database

        # Reconstruction
        self.reconstruction = Reconstruction()

        self.next_image_selector = NextImageSelector()

        self.correspondence_finder = CorrespondenceFinder()

        self.incremental_triangulator = IncrementalTriangulation()
        
        self.initializor = Initializor()


    # ----------------------------------------------------
    # Phase 1
    # ----------------------------------------------------

    def initialize(self):
        
        self.reconstruction = self.initializor.initialize(
        self.feature_database,
        self.camera_database,
        self.match_database
    )



    # ----------------------------------------------------
    # Phase 2
    # ----------------------------------------------------

    def register_next_image(self):

        image_id = self.next_image_selector.select(
            self.reconstruction,
            self.match_database
        )

        if image_id is None:
            return None

        object_points, image_points = self.correspondence_finder.find(
                self.reconstruction,
                image_id,
                self.match_database,
                self.feature_database
            )
        
        print("Object points:", len(object_points))
        print("Image points :", len(image_points))

        if len(object_points) < 6:
            return None

        print(object_points.shape)
        print(image_points.shape)

        print(object_points.dtype)
        print(image_points.dtype)

        print(self.camera_database[image_id]["K"])

        print(np.isnan(object_points).any())
        print(np.isnan(image_points).any())
        
        print("Unique object points:", len(np.unique(object_points, axis=0)))
        print("Unique image points :", len(np.unique(image_points, axis=0)))
        
        pose = PnPSolver.solve(
            object_points,
            image_points,
            self.camera_database[image_id]["K"]
        )
        print("Pose : ", pose)
        if pose is None:
            return None

        self.reconstruction.add_camera(
            pose["R"],
            pose["t"],
            image_id
        )

        print(
            f"Registered Image : {image_id}"
        )

        return image_id

    
    # ----------------------------------------------------
    # Phase 3
    # ----------------------------------------------------
    def triangulate_new_points(self, image_id):

        for match_set in self.match_database:

            ids = {
                match_set.image1_id,
                match_set.image2_id
            }

            if image_id not in ids:
                continue

            other = (
                match_set.image2_id
                if match_set.image1_id == image_id
                else match_set.image1_id
            )

            if other not in self.reconstruction.cameras:
                continue

            camera1 = self.reconstruction.cameras[
                match_set.image1_id
            ]

            camera2 = self.reconstruction.cameras[
                match_set.image2_id
            ]

            self.incremental_triangulator.triangulate(

                reconstruction=self.reconstruction,

                match_set=match_set,

                feature_database=self.feature_database,

                K1=self.camera_database[
                    match_set.image1_id
                ]["K"],

                K2=self.camera_database[
                    match_set.image2_id
                ]["K"],

                R1=camera1["R"],
                t1=camera1["t"],

                R2=camera2["R"],
                t2=camera2["t"]
            )


    # ----------------------------------------------------
    # Helper
    # ----------------------------------------------------

    def is_finished(self):

        """
        Returns True if all images have been registered.
        """

        return len(self.reconstruction.cameras) == len(
            self.feature_database
        )


    # ----------------------------------------------------
    # Main Pipeline
    # ----------------------------------------------------

    def run(self):

        print("=" * 60)
        print("INITIALIZATION")
        print("=" * 60)

        self.initialize()

        iteration = 1

        while not self.is_finished():

            print()
            print("=" * 60)
            print(f"ITERATION {iteration}")
            print("=" * 60)

            image_id = self.register_next_image()
            
            print("Next image:", image_id)

            if image_id is None:
                print("No more images can be registered.")
                break

            self.triangulate_new_points(image_id)

            print(f"Registered Cameras : {len(self.reconstruction.cameras)}")
            print(f"3D Points          : {len(self.reconstruction.points3D)}")
            print(f"Tracks             : {len(self.reconstruction.tracks)}")

            iteration += 1

        print()
        print("=" * 60)
        print("RECONSTRUCTION COMPLETE")
        print("=" * 60)

        print(f"Cameras : {len(self.reconstruction.cameras)}")
        print(f"Points  : {len(self.reconstruction.points3D)}")
        print(f"Tracks  : {len(self.reconstruction.tracks)}")

        return self.reconstruction