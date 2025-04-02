import numpy as np
import networkx as nx


def make_corrupt_network(adj_matrix, removal_prob, addition_prob):
    graph = nx.from_numpy_array(adj_matrix)
    edges = list(graph.edges())
    nodes = list(graph.nodes())

    num_remove = int(len(edges) * removal_prob)
    edges_to_remove = np.random.choice(len(edges), num_remove, replace=False)
    for idx in edges_to_remove:
        graph.remove_edge(*edges[idx])

    num_add = int(len(edges) * addition_prob)
    for _ in range(num_add):
        u, v = np.random.choice(nodes, 2, replace=False)
        if not graph.has_edge(u, v):
            graph.add_edge(u, v)

    # deleeting nodes with degree of freedom = 0
    degrees = dict(graph.degree())
    isolated_nodes = [node for node, deg in degrees.items() if deg == 0]
    for node in isolated_nodes:
        # add new edges
        possible_partners = [n for n in nodes if n != node and not graph.has_edge(node, n)]
        if possible_partners:
            partner = np.random.choice(possible_partners)
            graph.add_edge(node, partner)

    adj_matrix = nx.to_numpy_array(graph, nodelist=nodes)

    return adj_matrix
