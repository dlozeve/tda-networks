import graph_tool.all as gt


def find_cliques(G):
    """Returns all maximal cliques in an undirected graph.
    For each node *v*, a *maximal clique for v* is a largest complete
    subgraph containing *v*. The largest maximal clique is sometimes
    called the *maximum clique*.
    This function returns an iterator over cliques, each of which is a
    list of nodes. It is an iterative implementation, so should not
    suffer from recursion depth issues.
    Parameters
    ----------
    G : graph-tool graph
        An undirected graph.
    Returns
    -------
    iterator
        An iterator over maximal cliques, each of which is a list of
        nodes in `G`. The order of cliques is arbitrary.
    See Also
    --------
    find_cliques_recursive
        A recursive version of the same algorithm.
    Notes
    -----
    Taken from NetworkX.
    https://github.com/networkx/networkx/blob/master/networkx/algorithms/clique.py


    To obtain a list of all maximal cliques, use
    `list(find_cliques(G))`. However, be aware that in the worst-case,
    the length of this list can be exponential in the number of nodes in
    the graph (for example, when the graph is the complete graph). This
    function avoids storing all cliques in memory by only keeping
    current candidate node lists in memory during its search.
    This implementation is based on the algorithm published by Bron and
    Kerbosch (1973) [1]_, as adapted by Tomita, Tanaka and Takahashi
    (2006) [2]_ and discussed in Cazals and Karande (2008) [3]_. It
    essentially unrolls the recursion used in the references to avoid
    issues of recursion stack depth (for a recursive implementation, see
    :func:`find_cliques_recursive`).
    This algorithm ignores self-loops and parallel edges, since cliques
    are not conventionally defined with such edges.
    References
    ----------
    .. [1] Bron, C. and Kerbosch, J.
       "Algorithm 457: finding all cliques of an undirected graph".
       *Communications of the ACM* 16, 9 (Sep. 1973), 575--577.
       <http://portal.acm.org/citation.cfm?doid=362342.362367>
    .. [2] Etsuji Tomita, Akira Tanaka, Haruhisa Takahashi,
       "The worst-case time complexity for generating all maximal
       cliques and computational experiments",
       *Theoretical Computer Science*, Volume 363, Issue 1,
       Computing and Combinatorics,
       10th Annual International Conference on
       Computing and Combinatorics (COCOON 2004), 25 October 2006, Pages 28--42
       <https://doi.org/10.1016/j.tcs.2006.06.015>
    .. [3] F. Cazals, C. Karande,
       "A note on the problem of reporting maximal cliques",
       *Theoretical Computer Science*,
       Volume 407, Issues 1--3, 6 November 2008, Pages 564--568,
       <https://doi.org/10.1016/j.tcs.2008.05.010>
    """
    if len(G.get_vertices()) == 0:
        return

    adj = {u: {v for v in G.get_out_neighbors(u)} for u in G.vertices()}
    Q = [None]

    subg = set(G.get_vertices())
    cand = set(G.get_vertices())
    u = max(subg, key=lambda u: len(cand & adj[u]))
    ext_u = cand - adj[u]
    stack = []

    try:
        while True:
            if ext_u:
                q = ext_u.pop()
                cand.remove(q)
                Q[-1] = q
                adj_q = adj[q]
                subg_q = subg & adj_q
                if not subg_q:
                    yield Q[:]
                else:
                    cand_q = cand & adj_q
                    if cand_q:
                        stack.append((subg, cand, ext_u))
                        Q.append(None)
                        subg = subg_q
                        cand = cand_q
                        u = max(subg, key=lambda u: len(cand & adj[u]))
                        ext_u = cand - adj[u]
            else:
                Q.pop()
                subg, cand, ext_u = stack.pop()
    except IndexError:
        pass


def find_cliques_recursive(G):
    """Returns all maximal cliques in a graph.
    For each node *v*, a *maximal clique for v* is a largest complete
    subgraph containing *v*. The largest maximal clique is sometimes
    called the *maximum clique*.
    This function returns an iterator over cliques, each of which is a
    list of nodes. It is a recursive implementation, so may suffer from
    recursion depth issues.
    Parameters
    ----------
    G : graph-tool graph
    Returns
    -------
    iterator
        An iterator over maximal cliques, each of which is a list of
        nodes in `G`. The order of cliques is arbitrary.
    See Also
    --------
    find_cliques
        An iterative version of the same algorithm.
    Notes
    -----
    Taken from NetworkX.
    https://github.com/networkx/networkx/blob/master/networkx/algorithms/clique.py

    To obtain a list of all maximal cliques, use
    `list(find_cliques_recursive(G))`. However, be aware that in the
    worst-case, the length of this list can be exponential in the number
    of nodes in the graph (for example, when the graph is the complete
    graph). This function avoids storing all cliques in memory by only
    keeping current candidate node lists in memory during its search.
    This implementation is based on the algorithm published by Bron and
    Kerbosch (1973) [1]_, as adapted by Tomita, Tanaka and Takahashi
    (2006) [2]_ and discussed in Cazals and Karande (2008) [3]_. For a
    non-recursive implementation, see :func:`find_cliques`.
    This algorithm ignores self-loops and parallel edges, since cliques
    are not conventionally defined with such edges.
    References
    ----------
    .. [1] Bron, C. and Kerbosch, J.
       "Algorithm 457: finding all cliques of an undirected graph".
       *Communications of the ACM* 16, 9 (Sep. 1973), 575--577.
       <http://portal.acm.org/citation.cfm?doid=362342.362367>
    .. [2] Etsuji Tomita, Akira Tanaka, Haruhisa Takahashi,
       "The worst-case time complexity for generating all maximal
       cliques and computational experiments",
       *Theoretical Computer Science*, Volume 363, Issue 1,
       Computing and Combinatorics,
       10th Annual International Conference on
       Computing and Combinatorics (COCOON 2004), 25 October 2006, Pages 28--42
       <https://doi.org/10.1016/j.tcs.2006.06.015>
    .. [3] F. Cazals, C. Karande,
       "A note on the problem of reporting maximal cliques",
       *Theoretical Computer Science*,
       Volume 407, Issues 1--3, 6 November 2008, Pages 564--568,
       <https://doi.org/10.1016/j.tcs.2008.05.010>
    """
    if len(G.get_vertices()) == 0:
        return iter([])

    adj = {u: {v for v in G.get_out_neighbors(u)} for u in G.vertices()}
    Q = []

    def expand(subg, cand):
        u = max(subg, key=lambda u: len(cand & adj[u]))
        for q in cand - adj[u]:
            cand.remove(q)
            Q.append(q)
            adj_q = adj[q]
            subg_q = subg & adj_q
            if not subg_q:
                yield Q[:]
            else:
                cand_q = cand & adj_q
                if cand_q:
                    for clique in expand(subg_q, cand_q):
                        yield clique
            Q.pop()

    return expand(set(G.get_vertices()), set(G.get_vertices()))


def cliques_containing_node(G, nodes=None, cliques=None):
    """Returns a list of cliques containing the given node.
    Returns a single list or list of lists depending on input nodes.
    Optional list of cliques can be input if already computed.

    Taken from NetworkX.
    https://github.com/networkx/networkx/blob/master/networkx/algorithms/clique.py
    """
    if cliques is None:
        cliques = list(find_cliques(G))

    if nodes is None:
        nodes = list(G.get_vertices())   # none, get entire graph

    if not isinstance(nodes, list):   # check for a list
        v = nodes
        # assume it is a single value
        vcliques = [c for c in cliques if v in c]
    else:
        vcliques = {}
        for v in nodes:
            vcliques[v] = [c for c in cliques if v in c]
    return vcliques


if __name__ == "__main__":
    g = gt.collection.data["karate"]
    cliques = list(find_cliques(g))
    print(cliques)
    print(cliques_containing_node(g, 1))
