class InitialPairSelection:
    
    def __init__(self):
        pass
    
    def select(self, match_database):
        
        if len(match_database)==0:
            raise ValueError("Match database is empty.")
        
        best_match = None
        max_inliers = -1
        
        for matchset in match_database:
            num_inliers = len(matchset.matches)
            
            if num_inliers > max_inliers:
                max_inliers = num_inliers
                best_match = matchset
            
        return best_match