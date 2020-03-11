import networkx as nx
import numpy as np


G = nx.karate_club_graph()
graph_walker = GraphWalker(G)
walk_corpus = graph_walker.build_walk_corpus(walks_per_vertex=2, walk_length=5)
print(walk_corpus)
