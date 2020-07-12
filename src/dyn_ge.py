from static_ge import StaticGE
from utils.autoencoder import Autoencoder
import networkx as nx


def calculate_expand_size(prop_size=0.3, prev_input_dim=None, input_dim=None):
    return False, 0, 0


class DynGE(object):
    def __init__(self, graphs, embedding_dim, init_hidden_dims=None, v1=0.001, v2=0.001):
        super(DynGE, self).__init__()
        if init_hidden_dims is None:
            init_hidden_dims = []
        self.graphs = graphs
        self.graph_len = len(graphs)
        self.embedding_dim = embedding_dim
        self.init_hidden_dims = init_hidden_dims
        self.v1 = v1
        self.v2 = v2
        self.static_ges = []
        # self.models = []

    def get_all_embeddings(self):
        pass

    def get_embedding(self, index):
        if index < 0 or index >= self.graph_len:
            raise ValueError("index is invalid!")
        return self.static_ges[index].get_embedding()

    def train(self, prop_size=0.3, batch_size=64):
        model = Autoencoder(
            input_dim=len(self.graphs[0].nodes()),
            embedding_dim=self.embedding_dim,
            hidden_dims=self.init_hidden_dims,
            v1=self.v1,
            v2=self.v2
        )
        ge = StaticGE(G=self.graphs[0], model=model)
        ge.train(batch_size=batch_size)
        self.static_ges.append(ge)
        model = ge.get_model()

        for i in range(1, len(self.graphs)):
            graph = nx.Graph(self.graphs[i])
            prev_input_dim = len(self.graphs[i - 1].nodes())
            input_dim = len(graph.nodes())
            is_expand, add_node, add_layer = calculate_expand_size(prop_size, prev_input_dim, input_dim)
            if is_expand:
                pass

            ge = StaticGE(G=graph, model=model)
            ge.train(batch_size=batch_size)
            self.static_ges.append(ge)
            model = ge.get_model()


if __name__ == "__main__":
    # dy_ge = DynGE(graphs=graphs, embedding_dim=8, init_hidden_dims=[64, 16])
    pass
