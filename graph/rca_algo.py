"""
Groot RCA rank is a special version of PageRank algorithm based on our
use cases. Pagerank computes a ranking of the nodes in the graph G based on
the structure of the incoming links. It was originally designed as
an algorithm to rank web pages.

@author: hanzwang@ebay.com

PageRank References
----------
.. [1] A. Langville and C. Meyer,
   "A survey of eigenvector methods of web information retrieval."
   http://citeseer.ist.psu.edu/713792.html
.. [2] Page, Lawrence; Brin, Sergey; Motwani, Rajeev and Winograd, Terry,
   The PageRank citation ranking: Bringing order to the Web. 1999
   http://dbpubs.stanford.edu:8090/pub/showDoc.Fulltext?lang=en&doc=1999-66&format=pdf
"""

import networkx as nx
from sklearn import preprocessing
import numpy as np


def deep_rca_rank(G: nx.DiGraph, deep_rule, limited_start_set=None, max_iter=100, end_probability_weight=0.5,
                  end_weight_initial=1e-2, end_weight_anneal=2, use_entity=True,
                  use_start=True, use_self_rule=True, use_deep_rule=True, non_rule_weight=True,
                  weight='weight', data_name='unkown'):
    if len(G) == 0:
        return {}

    if not G.is_directed():
        D = G.to_directed()
    else:
        D = G

    def bfs(start_set):
        visited = set()
        current_list = []
        for n in start_set:
            current_list.append(n)
            visited.add(n)
        while len(current_list) > 0:
            n = current_list.pop()
            for nbr in D[n]:
                # if D[n][nbr]['weight'] > 0:
                if nbr not in visited:
                    current_list.append(nbr)
                    visited.add(nbr)
        print(f"bfs: {visited}")
        return list(visited)

    for u, v, d in D.edges(data='direction'):
        if d == 's':
            if use_self_rule:
                D[u][v]['weight'] = deep_rule.predict_likelihood(u, v, direction=d)
            else:
                D[u][v]['weight'] = 1.0
        elif d == 'd' or d == 'u':
            if use_deep_rule:
                D[u][v]['weight'] = deep_rule.predict_likelihood(u, v, direction=d)
            else:
                D[u][v]['weight'] = 1.0
        else:
            D[u][v]['weight'] = deep_rule.predict_likelihood(u, v, direction=d)
        if non_rule_weight:
            if 'non_rule_weight' in D[u][v]:
                if D[u][v]['non_rule_weight']:
                    D[u][v]['weight'] = 0.5 + D[u][v]['weight']
                    # print(f"{u}->{v} with {d} and {D[u][v]['weight']}")

    if limited_start_set is not None:
        limited_nodes = bfs(limited_start_set)
        D = D.subgraph(limited_nodes)

    start_probability = dict.fromkeys(D, 0)
    for n in limited_start_set:
        start_probability[n] = deep_rule.predict_start_probability(n)

    N = D.number_of_nodes()

    def normalize(d):
        sum_probability = 0
        for n in d:
            sum_probability = sum_probability + d[n]
        for n in d:
            d[n] = d[n] / max(1e-8, sum_probability)
        return d

    def forward(D, start_probability=None):
        in_probability = dict.fromkeys(D, 0)
        out_probability = dict.fromkeys(D, 0)

        for u, v, d in D.edges(data=weight):
            in_probability[v] = in_probability[v] + d
            out_probability[u] = out_probability[u] + d

        average_in_probability = 0
        average_out_probability = 0
        for n in D:
            average_in_probability = average_in_probability + in_probability[n]
            average_out_probability = average_out_probability + out_probability[n]

        if start_probability is None:
            start_probability = dict.fromkeys(D, 1.0 / N)
        start_probability = normalize(start_probability)

        # End Probability Setup
        if end_probability_weight is not None:
            average_out_probability = average_out_probability / N * end_probability_weight
        if average_out_probability == 0:
            average_out_probability = 1
        end_probability = dict.fromkeys(D, 0)
        for n in D:
            if end_probability_weight is None:
                end_probability[n] = out_probability[n] / N
            else:
                end_probability[n] = average_out_probability
        x = dict.copy(start_probability)
        res_probability = dict.fromkeys(D, 0)
        end_weight = end_weight_initial
        # power iteration: make up to max_iter iterations
        for _ in range(max_iter):
            xlast = x
            x = dict.fromkeys(xlast.keys(), 0)
            for n in D:
                sum_probability = end_probability[n] + out_probability[n]
                for nbr in D[n]:
                    x[nbr] += xlast[n] * D[n][nbr][weight] / sum_probability
            for n in D:
                sum_probability = end_probability[n] + out_probability[n]
                res_probability[n] += x[n] * end_probability[n] * end_weight / sum_probability
            x = deep_rule.self_conv(x)
            end_weight = min(1.0, end_weight * end_weight_anneal)
            # print('x_last:', xlast)
            # print('res:', res_probability)

        return normalize(res_probability)

    entity_weight = {}
    for n in D:
        entity_weight[n] = deep_rule.predict_entity_likelihood(n)
    entity_weight = normalize(entity_weight)

    end_probability = forward(D, start_probability if use_start else None)
    if use_entity:
        for n in D:
            end_probability[n] = end_probability[n] * entity_weight[n]
    end_probability = normalize(end_probability)

    if sum(end_probability.values()) == 0:
        if limited_start_set is not None:
            for n in limited_start_set:
                end_probability[n] = 1
            end_probability = normalize(end_probability)
    # print('entity_weight:')
    # print(entity_weight)
    # print('start_probability:')
    # print(start_probability)
    # print('end_probability:')
    # print(end_probability)
    return end_probability


def rca_rank(G, alpha=0.85, personalization=None,
             max_iter=100, tol=1.0e-6, nstart=None, weight='weight',
             dangling_process=True, neutralization=0.5, **kwargs):
    """Return the RCARank of the nodes in the graph.

    Parameters
    ----------
    G : graph
      A NetworkX directed graph of events. The directional link between events
      are created by event_graph_search or domain_graph_search and RCA rulebook.
      The links represent potential causalities.

    alpha : float, optional
      Damping parameter for PageRank, default=0.85.

    personalization: dict, optional
      The "personalization vector" consisting of a dictionary with a
      key for every graph node and nonzero personalization value for each node.
      By default, a uniform distribution is used.

    max_iter : integer, optional
      Maximum number of iterations in power method eigenvalue solver.

    tol : float, optional
      Error tolerance used to check convergence in power method solver.

    nstart : dictionary, optional
      Starting value of PageRank iteration for each node.

    weight : key, optional
      Edge data key to use as weight.  If None weights are set to 1.

    dangling_process: boolean, optional
      The outedges to be assigned to any "dangling" nodes, i.e., nodes without
      any outedges. This process generates a dict of weights for dangling node
      (1) and non-dangling node (neutralization). This will balance the impact
      of each iteration with more "focus" of the dangling nodes.

      By default, dangling nodes are given outedges according to the
      personalization vector (uniform if not specified).
      This must be selected to result in an irreducible transition
      matrix (see notes under google_matrix). It may be common to have the
      dangling dict to be the same as the personalization dict.

    neutralization: float, optional
      This is the weight parameter of non-dangling node.
      The dangling nodes weight is 1.


    Returns
    -------
    rcarank : dictionary
       Dictionary of nodes with PageRank as value

    Examples
    --------
    >>> G = nx.DiGraph(nx.path_graph(4))
    >>> pr = rca_rank(G, dangling_process = False)

    Notes
    -----
    The eigenvector calculation is done by the power iteration method
    and has no guarantee of convergence.  The iteration will stop
    after max_iter iterations or an error tolerance of
    number_of_nodes(G)*tol has been reached.

    The PageRank algorithm was designed for directed graphs but this
    algorithm does not check if the input graph is directed and will
    execute on undirected graphs by converting each edge in the
    directed graph to two edges.
    """
    if len(G) == 0:
        return {}

    if not G.is_directed():
        D = G.to_directed()
    else:
        D = G

    # Create a copy in (right) stochastic form
    W = nx.stochastic_graph(D, weight=weight)
    N = W.number_of_nodes()

    # Choose fixed starting vector if not given
    if nstart is None:
        x = dict.fromkeys(W, 1.0 / N)
    else:
        # Normalized nstart vector
        s = float(sum(nstart.values()))
        x = dict((k, v / s) for k, v in nstart.items())

    if personalization is None:
        # Assign uniform personalization vector if not given
        p = dict.fromkeys(W, 1.0 / N)
    else:
        s = float(sum(personalization.values()))
        p = dict((k, v / s) for k, v in personalization.items())

    if not dangling_process:
        # Use personalization vector if dangling vector not specified
        dangling_weights = p
    else:
        dangling = {}
        for node in G:
            if G[node]:
                dangling[node] = neutralization
            else:
                dangling[node] = 1
        s = float(sum(dangling.values()))
        dangling_weights = dict((k, v / s) for k, v in dangling.items())
    dangling_nodes = [n for n in W if W.out_degree(n, weight=weight) == 0.0]

    # power iteration: make up to max_iter iterations
    for _ in range(max_iter):
        xlast = x
        x = dict.fromkeys(xlast.keys(), 0)
        danglesum = alpha * sum(xlast[n] for n in dangling_nodes)
        for n in x:
            # this matrix multiply looks odd because it is
            # doing a left multiply x^T=xlast^T*W
            for nbr in W[n]:
                x[nbr] += alpha * xlast[n] * W[n][nbr][weight]
            x[n] += danglesum * dangling_weights.get(n, 0) + (1.0 - alpha) * p.get(n, 0)
        # check convergence, l1 norm
        err = sum([abs(x[n] - xlast[n]) for n in x])
        if err < N * tol:
            return x
    raise nx.PowerIterationFailedConvergence(max_iter)


if __name__ == '__main__':
    # This is an example

    G = nx.DiGraph()
    pages = ["1", "2", "3", "4", "5"]
    G.add_nodes_from(pages)
    G.add_edges_from(
        [('1', '2'), ('1', '4'), ('2', '3'), ('2', '4'), ('4', '5')])
    # G.add_weighted_edges_from(
    # [('1', '2', 0.5), ('1', '4', 1), ('2', '3', 1), ('3', '2', 1), ('2', '4', 1 ), ('4', '5', 1) ])
    # [('1', '2', 0.1), ('2', '3', 0.1), ('2', '4', 0.1), ('5', '6', 1)])
    pr = rca_rank(G, dangling_process=True)
    min_max_scaler = preprocessing.MinMaxScaler()
    result_nparra = np.array([[v] for _, v in pr.items()])
    x_scaled_p = min_max_scaler.fit_transform(result_nparra)
    print(x_scaled_p)