import numpy as np
import dionysus as d


def diagram_array(dgm):
    res = []
    for p in dgm:
        if p.death != np.inf:
            res.append([p.birth, p.death])
    return np.array(res)


def SW_approx(dgm1, dgm2, M):
    dgm1 = diagram_array(dgm1)
    dgm2 = diagram_array(dgm2)
    # Add \pi_\delta(dgm1) to dgm2 and vice-versa
    proj1 = dgm1.dot([1, 1])/np.sqrt(2)
    proj2 = dgm2.dot([1, 1])/np.sqrt(2)
    dgm1 = np.vstack((dgm1, np.vstack((proj2, proj2)).T))
    dgm2 = np.vstack((dgm2, np.vstack((proj1, proj1)).T))
    SW = 0
    theta = -np.pi/2
    s = np.pi/M
    for i in range(M):
        # Project each diagram on the direction theta
        vec = [1, np.arctan(theta)]
        vec = vec / np.linalg.norm(vec)
        V1 = dgm1.dot(vec)
        V2 = dgm2.dot(vec)
        # Sort the projections
        V1.sort()
        V2.sort()
        # l1-distance between the projections
        SW = SW + s * np.sum(np.abs(V1 - V2))
        theta = theta + s
    return 1/np.pi * SW
