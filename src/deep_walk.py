from time import time

import networkx as nx
from gensim.models import Word2Vec
from gensim.test.utils import get_tmpfile
from joblib import cpu_count

from src.utils.graph_walker import GraphWalker
from src.utils.visualize import plot_embeddings


class SkipGram(Word2Vec):
    def __init__(self, sentences, embedding_size, window_size):
        """

        :param sentences:
        :param embedding_size:
        :param window_size:

        Default params:
        LINK: https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec
        min_count (int, optional) – Ignores all words with total frequency lower than this.
        workers (int, optional) – Use these many worker threads to train the model (=faster training with multicore machines).
        sg ({0, 1}, optional) – Training algorithm: 1 for skip-gram; otherwise CBOW.
        hs ({0, 1}, optional) – If 1, hierarchical softmax will be used for model training. If 0, and negative is non-zero, negative sampling will be used.
        """
        # TODO: add more parameter here

        super(SkipGram, self).__init__(sentences=sentences, size=embedding_size, window=window_size, sg=1, hs=1,
                                       min_count=0, workers=cpu_count())


class DeepWalk():
    def __init__(self, graph, walks_per_vertex, walk_length):
        """

        :param graph: graph input
        :param walks_per_vertex: (gamma param in paper) the number of looping through over all of nodes in graph
        :param walk_length: length of each walk path start node u
        """
        self.graph = graph
        self.graph_walker = GraphWalker(self.graph)
        self.walks_corpus = self.graph_walker.build_walk_corpus(walks_per_vertex, walk_length)

    def train(self, embedding_size, window_size):
        print("Start training...")
        start_time = time()
        self.model = SkipGram(self.walks_corpus, embedding_size, window_size)

        finish_time = time()
        print("Done! Training time: ", finish_time - start_time)

    def get_embedding(self):
        self.embedding = {}
        for node in list(self.graph.nodes()):
            self.embedding[str(node)] = self.model.wv[str(node)]

        return self.embedding

    def save_model(self, path=None):
        if path is None:
            path = get_tmpfile("node_vectors.kv")
        self.model.wv.save(path)


if __name__ == "__main__":
    # Change this to run test
    test_num = 1

    # Test 1
    if test_num == 1:
        G = nx.read_edgelist('../data/Wiki_edgelist.txt',
                             create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])

        model = DeepWalk(G, walks_per_vertex=20, walk_length=10)
        model.train(embedding_size=128, window_size=5)
        embeddings = model.get_embedding()
        plot_embeddings(G, embeddings, path_file="../data/Wiki_category.txt")

    # Test 2
    if test_num == 2:
        G = nx.karate_club_graph()

        model = DeepWalk(G, walks_per_vertex=20, walk_length=10)
        model.train(embedding_size=128, window_size=5)
        embeddings = model.get_embedding()
        plot_embeddings(G, embeddings)
