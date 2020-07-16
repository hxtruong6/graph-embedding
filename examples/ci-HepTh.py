import numpy as np
import networkx as nx
import os
from os import listdir
from os.path import isfile, join
import re
from time import time
from dyn_ge import DynGE

# Dataset link: https://snap.stanford.edu/data/cit-HepTh.html
from static_ge import StaticGE
from utils.link_prediction import preprocessing_graph, run_evaluate, run_predict, plot_link_prediction_graph


def handle_citHepTH_dataset(edge_list_path=None, abstract_path=None, verbose=False):
    if edge_list_path is None or abstract_path is None:
        raise ValueError("Must be provide path of dataset")

    print("Reading Cit-HepTH dataset...")
    begin_time = time()
    G = nx.read_edgelist(edge_list_path, nodetype=int)
    V = G.nodes()

    year_folder = os.listdir(abstract_path)
    abs_nodes_dic = []
    abs_nodes = []
    nodes_by_year = {}
    for y in sorted(year_folder):
        nodes_by_year[y] = []
        curr_path = join(abstract_path, y)
        files = [f for f in listdir(curr_path) if isfile(join(curr_path, f))]
        for file in files:
            v = int(file.strip().split('.')[0])
            if v not in V:
                continue
            abs_nodes.append(v)
            nodes_by_year[y].append(v)
            # file format: '9205018.abs'
            # with open(join(curr_path, file)) as fi:
            #     content = fi.read()
            #     contents = re.split(r'(\\)+', content)
            #     abs_nodes_dic.append({v: {"info": contents[2], "abstract": contents[4]}})
        # print(f"Year {y}: number of nodes: {len(nodes_by_year[y])}")

    graphs = []
    years = list(nodes_by_year.keys())
    prev_year = None
    for i, year in enumerate(reversed(years)):
        graph = None
        if i == 0:
            graph = G.copy()
        else:
            graph = graphs[i - 1].copy()
            graph.remove_nodes_from(nodes_by_year[prev_year])
        if verbose:
            print(f"Year {year}: |V|={len(graph.nodes())}\t |E|={len(graph.edges())}")
        prev_year = year
        graphs.append(graph)

    print(f"Reading in {round(time() - begin_time, 2)}s. Done!")
    return list(reversed(graphs))


def get_ciHepTH_dataset():
    return handle_citHepTH_dataset(
        edge_list_path="../data/cit-HepTh/cit-HepTh.txt",
        abstract_path="../data/cit-HepTh/cit-HepTh-abstracts/"
    )


def ciHepTH_link_prediction(G: nx.Graph, top_k=10):
    # unconnected_egdes =x
    G_df, G_partial = preprocessing_graph(G=G)
    # TODO: check remove maximum 15% omissible egde in total egdes.
    ge = StaticGE(G=G_partial, embedding_dim=4, hidden_dims=[8])
    ge.train(epochs=100, skip_print=20, learning_rate=0.001)
    embedding = ge.get_embedding()

    link_pred_model = run_evaluate(G_df, embedding, num_boost_round=2000)
    possible_egdes_df = G_df[G_df['link'] == 0]
    y_pred = run_predict(data=possible_egdes_df, embedding=embedding, model=link_pred_model)

    # get top K link prediction
    # sorted_y_pred, sorted_possible_edges = zip(*sorted(zip(y_pred, possible_egdes)))
    node_1 = possible_egdes_df['node_1'].to_list()
    node_2 = possible_egdes_df['node_2'].to_list()

    # unconnected_edges = [possible_egdes_df['node_1'].to_list(), possible_egdes_df['node_2'].to_list()]
    unconnected_edges = [(node_1[i], node_2[i]) for i in range(len(node_1))]
    sorted_y_pred, sorted_possible_edges = (list(t) for t in
                                            zip(*sorted(
                                                zip(y_pred, unconnected_edges),
                                                reverse=True)
                                                ))

    print(f"Top {top_k} predicted edges: edge|accuracy")
    for i in range(top_k):
        print(f"{sorted_possible_edges[i]} : {sorted_y_pred[i]}")

    plot_link_prediction_graph(G=G, pred_edges=sorted_possible_edges[:top_k])
    # plot_link_prediction_graph(G=G, pred_edges=sorted_possible_edges[:top_k], pred_acc=sorted_y_pred[:top_k])


if __name__ == "__main__":
    # graphs = handle_citHepTH_dataset(
    #     edge_list_path="../data/cit-HepTh/cit-HepTh.txt",
    #     abstract_path="../data/cit-HepTh/cit-HepTh-abstracts/",
    #     verbose=False
    # )
    # print(len(graphs))
    # graphs = graphs[:2]
    #
    # dy_ge = DynGE(graphs=graphs, embedding_dim=4, init_hidden_dims=[64, 16])
    # dy_ge.train()
    # embeddings = dy_ge.get_all_embeddings()
    # for e in embeddings:
    #     print(embeddings[:5])

    g = nx.karate_club_graph().to_undirected()
    ciHepTH_link_prediction(G=g)
