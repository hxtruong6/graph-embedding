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


if __name__ == "__main__":
    graphs = handle_citHepTH_dataset(
        edge_list_path="../data/cit-HepTh/cit-HepTh.txt",
        abstract_path="../data/cit-HepTh/cit-HepTh-abstracts/",
        verbose=False
    )
    # print(len(graphs))
    graphs = graphs[:2]

    dy_ge = DynGE(graphs=graphs, embedding_dim=4, init_hidden_dims=[64, 16])
    dy_ge.train()
    embeddings = dy_ge.get_all_embeddings()
    for e in embeddings:
        print(embeddings[:5])
