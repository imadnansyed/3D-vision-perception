import numpy as np


class CheiralityCheck:

    def __init__(self):
        pass


    def triangulate_points(self, Pl, Pr, xl, xr):

        A = np.array([

            xl[0] * Pl[2] - Pl[0],
            xl[1] * Pl[2] - Pl[1],

            xr[0] * Pr[2] - Pr[0],
            xr[1] * Pr[2] - Pr[1]

        ])

        _, _, Vt = np.linalg.svd(A)

        X = Vt[-1]

        X = X / X[3]

        return X[:3]



    def check(self, pts1, pts2, Kl, Kr, E):
        
        U, S, Vt = np.linalg.svd(E)

        W = np.array([[0, -1, 0],
                    [1,  0, 0],
                    [0,  0, 1]])

        R1 = U @ W @ Vt
        R2 = U @ W.T @ Vt

        if np.linalg.det(R1) < 0:
            R1 *= -1
        if np.linalg.det(R2) < 0:
            R2 *= -1

        t1 = U[:, 2]
        t2 = -U[:, 2]

        poses = [(R1, t1), (R1, t2), (R2, t1), (R2, t2)]

        best_pose = None
        max_positive_depth = 0

        Pl = Kl @ np.hstack((
            np.eye(3),
            np.zeros((3, 1))
        ))

        for R, t in poses:

            Pr = Kr @ np.hstack((
                R,
                t.reshape(3, 1)
            ))

            positive_count = 0

            num_samples = min(20, len(pts1))

            for i in range(num_samples):

                X = self.triangulate_points(
                    Pl,
                    Pr,
                    pts1[i],
                    pts2[i]
                )

                depth_left = X[2]

                X_right = R @ X + t


                depth_right = X_right[2]


                if depth_left > 0 and depth_right > 0:
                    positive_count += 1

            if positive_count > max_positive_depth:

                max_positive_depth = positive_count
                best_pose = (R, t)

        R_final, t_final = best_pose
        
        return R_final, t_final