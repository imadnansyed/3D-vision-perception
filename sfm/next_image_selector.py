class NextImageSelector:
    @staticmethod
    def select(reconstruction, match_database):
        registered = set(reconstruction.cameras.keys())
        
        scores = {}
    
        for match in match_database:
            
            i = match.image1_id
            j = match.image2_id
            
            n = len(match.matches)
            
            # i is registered and j in not
            if i in registered and j not in registered:
                scores[j] = scores.get(j,0) + n 
                
            # j is registered and i is not 
            elif j in registered and i not in registered:
                scores[i] = scores.get(i, 0) + n
            
        if len(scores) == 0:
            return None
        
        print("Selecting next image...")
        print(max(scores, key=scores.get))
        
        return max(scores, key=scores.get)