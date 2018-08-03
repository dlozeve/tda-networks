import numpy as np
import igraph as ig
import dionysus as d

import pickle
import matplotlib.pyplot as plt

plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = 10, 6


def sliding_windows(g, res=0.1, overlap=0):
    times = np.array(g.es["time"])
    duration = res * (times.max() - times.min())
    windows = []
    for i in range(int(1/res)-1):
        edges = g.es.select(time_gt=times.min() + duration*i,
                            time_lt=times.min() + duration*(i+1))
        windows.append(g.subgraph_edges(edges))
    return windows


def max_simplicial_complex(g):
    return d.Filtration(g.maximal_cliques())


def find_transitions(a):
    res = []
    prev = False
    for i, cur in enumerate(a):
        if cur != prev:
            res.append(i)
        prev = cur
    return res


def presence_times(g):
    max_simplicial_complex = d.Filtration(g.cliques())
    filts = []
    for t in np.sort(np.unique(g.es["time"])):
        edges = g.es.select(time_eq=t)
        cliques = g.subgraph_edges(edges).cliques()
        filts.append(d.Filtration(cliques))
    presences = [[s in filt for filt in filts] for s in max_simplicial_complex]
    presences = [find_transitions(p) for p in presences]
    return (max_simplicial_complex, presences)


if __name__ == "__main__":
    # Import the data
    g = ig.read("data/sociopatterns/infectious/infectious.graphml")
    print(g.summary())
    # Segment the network into sliding windows (resolution = 5%)
    wins = sliding_windows(g, 0.05)
    # Compute the presence times of maximal simplices for an example window
    print(wins[0].summary())
    (f, t) = presence_times(wins[0])
    for s in f:
        print(s)
    print(t)
    # Compute the zigzag homology on the window
    print("Computing zigzag persistence...")
    zz, dgms, cells = d.zigzag_homology_persistence(f, t)
    for i, dgm in enumerate(dgms):
        print("Dimension: {}".format(i))
        for p in dgm:
            print(p)
    # pickle.dump(dgms, open("diagrams.p", "wb"))
    # Plot the persistence diagrams
    # for i, dgm in enumerate(dgms):
    #     d.plot.plot_diagram(dgm, show=False)
    #     plt.savefig("dgm_{}.png".format(i))
