def generate_pairs(features, window=3):

    pairs = []

    n = len(features)

    for i in range(n):

        for j in range(i + 1, min(i + window + 1, n)):

            pairs.append((features[i], features[j]))

    return pairs