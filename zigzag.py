#!/usr/bin/env python3

import numpy as np
import igraph as ig
import dionysus as d


def sliding_windows(g, res=0.1, overlap=0):
    """Compute subnetworks of a temporal network based on temporal
    partitioning of the time range.

    :param g: igraph Graph
    :param res: resolution
    :param overlap: overlap

    :return: a list of temporal networks.
    """
    times = np.array(g.es["time"])
    duration = res * (times.max() - times.min())
    windows = []
    for i in range(int(1/res)):
        edges = g.es.select(time_gt=times.min() + duration*i,
                            time_lt=times.min() + duration*(i+1))
        windows.append(g.subgraph_edges(edges))
    return windows


def max_simplicial_complex(g):
    """Return the maximal simplicial complex of a network g.
    """
    return d.Filtration(g.maximal_cliques())


def find_transitions(a):
    """Find the transition times in an array of presence times.
    """
    res = []
    prev = False
    for i, cur in enumerate(a):
        if cur != prev:
            res.append(i)
        prev = cur
    return res


def presence_times(g):
    """Compute the data required to compute zigzag persistence:
    simplicial complex and transition times.

    :param g: igraph Graph

    :return: a tuple with the maximum simplicial complex and the
             transition times of each simplex.
    """
    max_simplicial_complex = d.Filtration(g.cliques())
    filts = []
    for t in np.sort(np.unique(g.es["time"])):
        edges = g.es.select(time_eq=t)
        cliques = g.subgraph_edges(edges).cliques()
        filts.append(d.Filtration(cliques))
    presences = [[s in filt for filt in filts] for s in max_simplicial_complex]
    presences = [find_transitions(p) for p in presences]
    return (max_simplicial_complex, presences)


def zigzag_network(g):
    """Compute zigzag persistence on a temporal network.

    :param g: igraph Graph

    :return: a list of persistence diagrams.
    """
    (f, t) = presence_times(g)
    _, dgms, _ = d.zigzag_homology_persistence(f, t)
    return dgms
