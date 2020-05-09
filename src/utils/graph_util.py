import networkx as nx


def print_graph_stats(G):
    print('# of nodes: %d, # of edges: %d' % (G.number_of_nodes(), G.number_of_edges()))


def preprocess_graph(graph):
    graph = nx.Graph(graph)
    node2idx = {}
    idx2node = []
    node_size = 0
    for node in graph.nodes():
        node2idx[node] = node_size
        idx2node.append(node)
        node_size += 1
    return idx2node, node2idx
