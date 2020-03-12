import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


class GraphWalker:
    def __init__(self, G):
        self.G = G
        self.walk_corpus = []

    def random_walk(self, start_node, walk_length):
        walk_path = [start_node]
        while len(walk_path) < walk_length:
            curr_node = walk_path[-1]  # get last node in walk path
            neighbor_nodes = list(self.G.neighbors(curr_node))
            if len(neighbor_nodes) > 0:
                next_node = np.random.choice(neighbor_nodes)
                walk_path.append(next_node)
            else:  # No any neighbor at this vertex
                walk_path.append(curr_node)
        return [str(node) for node in walk_path]

    def build_walk_corpus(self, walks_per_vertex, walk_length):
        shuffle_nodes = list(self.G.nodes())
        for _ in range(walks_per_vertex):
            np.random.shuffle(shuffle_nodes)  # Shuffle nodes in graph
            # print(shuffle_nodes)
            for node in shuffle_nodes:
                walk_path = self.random_walk(node, walk_length)
                # print(walk_path)
                self.walk_corpus.append(walk_path)
        return self.walk_corpus


if __name__ == "__main__":
    # G = nx.read_edgelist("./Wiki_edgelist.txt", data=(('weight',int),))
    G = nx.karate_club_graph()
    # nx.draw(G)  # networkx draw()
    # plt.draw()  # pyplot draw()
    graph_walker = GraphWalker(G)
    walk_corpus = graph_walker.build_walk_corpus(walks_per_vertex=2, walk_length=5)
    print(walk_corpus)
