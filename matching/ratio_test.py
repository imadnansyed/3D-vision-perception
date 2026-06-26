class RatioTest:
    def __init__(self, threshold = 0.75):
        self.threshold = threshold
        
    def test(self, matches):
        good_matches = []
        
        for m, n in matches:
            if m.distance < self.threshold * n.distance:
                good_matches.append(m)
                
        return good_matches