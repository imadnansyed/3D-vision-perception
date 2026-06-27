from features.image_loader import ImageLoader
from features.features_extractor import FeaturesExtractor

from matching.matcher import Matcher
from matching.pair_generator import generate_pairs
from matching.ratio_test import RatioTest
from matching.geometric_verification import GeometricVerification
from sfm.decompose_p import DecomposeP
from sfm.essential_matrix import EssentialMatrix
from sfm.initial_pair_selection import InitialPairSelection
from sfm.pose_recovery import CheiralityCheck



def main():
    
    dateset_path = "images"
    
    images = ImageLoader(dateset_path).load_images()
    
    extractor = FeaturesExtractor()
    
    features_database = extractor.extract(images)
    
    print("=" * 60)
    print("PHASE 1 : FEATURE EXTRACTION")
    print("=" * 60)

    # for feature in features_database:

    #     print(
    #         f"{feature.image_name:<20}"
    #         f" Keypoints : {len(feature.keypoints)}"
    #     )
    
    # **************  Making pairs *************************
    
    pairs = generate_pairs(features_database)
    
    matcher = Matcher()
    
    ratio_test = RatioTest()
    
    verifier = GeometricVerification()
    
    match_database = []
    
    print("\n")
    print("=" * 60)
    print("PHASE 2 : FEATURE MATCHING")
    print("=" * 60)
    
    for feature1, feature2 in pairs:
        matches = matcher.match(feature1.descriptors, feature2.descriptors)
        
        good_matches = ratio_test.test(matches=matches)
        
        # print(f"Raw Matches      : {len(matches)}")
        # print(f"After Ratio Test : {len(good_matches)}")
        
        match_set = verifier.verify(
            feature1, feature2, good_matches
        )

        # print(f"RANSAC Inliers   : {len(match_set.matches)}")
        
        match_database.append(match_set)
        
    

    # print("\n")
    # print("=" * 60)
    # print("SUMMARY")
    # print("=" * 60)

    # print(f"Images          : {len(features_database)}")
    # print(f"Image Pairs     : {len(pairs)}")
    # print(f"Verified Pairs  : {len(match_database)}")
    
    print("\n" + "=" * 60)
    print("PHASE 3 : INITIALIZATION")
    print("=" * 60)
    
    # extract Camera Matrices from the dino.mat file => will use only K for each camera
    camera_database = DecomposeP.get_camera_matrices()
    
    # initial pair selection
    selector = InitialPairSelection()
    initial_pair = selector.select(match_database)
    
    print(f"Initial Image Pair : "
      f"{initial_pair.image1_id} <-> {initial_pair.image2_id}")

    print(f"Verified Matches   : {len(initial_pair.matches)}")
    
    
    K1 = camera_database[initial_pair.image1_id]["K"]
    K2 = camera_database[initial_pair.image2_id]["K"]
    
    # compute Essential matrix
    
    E = EssentialMatrix.compute(initial_pair.F, K1, K2)
    
    # recover pose
    checker = CheiralityCheck()
    R, t = checker.check(initial_pair.pts1, initial_pair.pts2, K1, K2, E)
    
    
    

        
if __name__ == "__main__":
    main()