from features.image_loader import ImageLoader
from features.features_extractor import FeaturesExtractor

from matching.matcher import Matcher
from matching.pair_generator import generate_pairs
from matching.ratio_test import RatioTest
from matching.geometric_verification import GeometricVerification

from sfm.decompose_p import DecomposeP
from sfm.incremental_sfm import IncrementalSfM


def main():

    # ==========================================================
    # PHASE 1 : LOAD IMAGES
    # ==========================================================

    dataset_path = "images"

    images = ImageLoader(dataset_path).load_images()

    print("=" * 60)
    print("PHASE 1 : FEATURE EXTRACTION")
    print("=" * 60)

    extractor = FeaturesExtractor()

    feature_database = extractor.extract(images)

    print(f"Images Loaded      : {len(images)}")
    print(f"Features Extracted : {len(feature_database)}")

    # ==========================================================
    # PHASE 2 : FEATURE MATCHING
    # ==========================================================

    print()
    print("=" * 60)
    print("PHASE 2 : FEATURE MATCHING")
    print("=" * 60)

    pairs = generate_pairs(feature_database)

    matcher = Matcher()
    ratio_test = RatioTest()
    verifier = GeometricVerification()

    match_database = []

    for feature1, feature2 in pairs:

        matches = matcher.match(
            feature1.descriptors,
            feature2.descriptors
        )

        good_matches = ratio_test.test(matches)

        match_set = verifier.verify(
            feature1,
            feature2,
            good_matches
        )

        if match_set is None:
            continue

        match_database.append(match_set)

        # print(
        #     f"{feature1.image_id:02d} <-> {feature2.image_id:02d}   "
        #     f"Inliers : {len(match_set.matches)}"
        # )

    print()
    print(f"Verified Image Pairs : {len(match_database)}")

    # ==========================================================
    # PHASE 3 : CAMERA DATABASE
    # ==========================================================

    print()
    print("=" * 60)
    print("PHASE 3 : CAMERA PARAMETERS")
    print("=" * 60)

    camera_database = DecomposeP.get_camera_matrices()

    print(f"Cameras Loaded : {len(camera_database)}")

    # ==========================================================
    # PHASE 4 : INCREMENTAL SFM
    # ==========================================================

    print()
    print("=" * 60)
    print("PHASE 4 : INCREMENTAL STRUCTURE FROM MOTION")
    print("=" * 60)

    sfm = IncrementalSfM(
        feature_database=feature_database,
        camera_database=camera_database,
        match_database=match_database
    )

    reconstruction = sfm.run()

    # ==========================================================
    # SUMMARY
    # ==========================================================

    print()
    print("=" * 60)
    print("FINAL RECONSTRUCTION")
    print("=" * 60)

    print(f"Registered Cameras : {len(reconstruction.cameras)}")
    print(f"3D Points          : {len(reconstruction.points3D)}")
    print(f"Tracks             : {len(reconstruction.tracks)}")


if __name__ == "__main__":
    main()
    