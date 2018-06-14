import numpy as np
import igraph as ig
import dionysus as d


def wrcf(G, weight="weight"):
    # Define filtration step 0 as the set of all nodes
    filt = d.Filtration()
    for v in G.vs:
        filt.append(d.Simplex([v.index], 0))
    # Rank all edge weights
    distinct_weights = np.unique(G.es[weight])[::-1]
    for t, w in enumerate(distinct_weights):
        # At filtration step t, threshold the graph at weight[t]
        subg = G.subgraph_edges(G.es(lambda e: e[weight] >= w))
        # Find all maximal cliques and define them to be simplices
        for clique in subg.maximal_cliques():
            for s in d.closure([d.Simplex(clique)], len(clique)):
                filt.append(d.Simplex(s, t+1))
    filt.sort()
    return(filt)
