import networkx as nx
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import lightgbm as lgbm
import pandas as pd
import matplotlib.pyplot as plt


# from static_ge import StaticGE


def get_unconnected_pairs(G: nx.Graph):
    # TODO: convert to sparse matrix
    node_list = list(G.nodes())
    adj_G = nx.adj_matrix(G)

    # get unconnected node-pairs
    all_unconnected_pairs = []

    # traverse adjacency matrix. find all unconnected node with maximum 2nd order
    offset = 0
    for i in tqdm(range(adj_G.shape[0])):
        for j in range(offset, adj_G.shape[1]):
            if i != j:
                if nx.shortest_path_length(G, i, j) <= 2:
                    if adj_G[i, j] == 0:
                        all_unconnected_pairs.append([node_list[i], node_list[j]])

        offset = offset + 1

    return all_unconnected_pairs


def run_evaluate(data, embedding, alg=None, num_boost_round=1000, early_stopping_rounds=20):
    if alg == "Node2Vec":
        x = [(embedding[str(i)] + embedding[str(j)]) for i, j in zip(data['node_1'], data['node_2'])]
    else:
        x = [(embedding[i] + embedding[j]) for i, j in zip(data['node_1'], data['node_2'])]

    # TODO: check unbalance dataset
    X_train, X_test, y_train, y_test = train_test_split(
        np.array(x),
        data['link'],
        test_size=0.3,
        random_state=35,
        stratify=data['link']
    )

    train_data = lgbm.Dataset(X_train, y_train)
    test_data = lgbm.Dataset(X_test, y_test)

    # define parameters
    parameters = {
        'objective': 'binary',
        'metric': 'auc',
        'is_unbalance': 'true',
        'feature_fraction': 0.5,
        'bagging_fraction': 0.5,
        'bagging_freq': 20,
        'num_threads': 2,
        'seed': 76,
        'verbosity': -1
    }

    # train lightGBM model
    model = lgbm.train(parameters,
                       train_data,
                       valid_sets=test_data,
                       num_boost_round=100000,
                       early_stopping_rounds=20,
                       verbose_eval=10,
                       )

    y_pred = model.predict(X_test)
    print("#----\nROC AUC Score: ", roc_auc_score(y_test, y_pred, average=None))
    # roc_curve(y_test, y_pred)
    return model


def run_predict(data, embedding, model):
    x = [(embedding[i] + embedding[j]) for i, j in zip(data['node_1'], data['node_2'])]
    y_pred = model.predict(x)
    return y_pred


# https://www.analyticsvidhya.com/blog/2020/01/link-prediction-how-to-predict-your-future-connections-on-facebook/
def link_predict_evaluate(G: nx.Graph):
    node_list_1 = []
    node_list_2 = []

    for u, v in tqdm(G.edges):
        node_list_1.append(u)
        node_list_2.append(v)

    fb_df = pd.DataFrame({'node_1': node_list_1, 'node_2': node_list_2})

    all_unconnected_pairs = get_unconnected_pairs(G)

    node_1_unlinked = [i[0] for i in all_unconnected_pairs]
    node_2_unlinked = [i[1] for i in all_unconnected_pairs]

    data = pd.DataFrame({'node_1': node_1_unlinked,
                         'node_2': node_2_unlinked})

    # add target variable 'link'
    data['link'] = 0

    initial_node_count = len(G.nodes)

    fb_df_temp = fb_df.copy()

    # empty list to store removable links
    omissible_links_index = []

    for i in tqdm(fb_df.index.values):

        # remove a node pair and build a new graph
        G_temp = nx.from_pandas_edgelist(fb_df_temp.drop(index=i), "node_1", "node_2", create_using=nx.Graph())

        # check there is no spliting of graph and number of nodes is same
        if (nx.number_connected_components(G_temp) == 1) and (len(G_temp.nodes) == initial_node_count):
            omissible_links_index.append(i)
            fb_df_temp = fb_df_temp.drop(index=i)

    # create dataframe of removable edges
    fb_df_ghost = fb_df.loc[omissible_links_index]

    # add the target variable 'link'
    fb_df_ghost['link'] = 1

    data = data.append(fb_df_ghost[['node_1', 'node_2', 'link']], ignore_index=True)

    # drop removable edges
    fb_df_partial = fb_df.drop(index=fb_df_ghost.index.values)

    # build graph
    G_data = nx.from_pandas_edgelist(fb_df_partial, "node_1", "node_2", create_using=nx.Graph())
    #
    # # # SDNE
    # ge = StaticGE(G=G_data, embedding_dim=4, hidden_dims=[8])
    # ge.train(epochs=40)
    # embedding = ge.get_embedding()
    #
    # run_evaluate(data, embedding)
    #
    # # Node2vec
    # node2vec_model = Node2Vec(G_data, dimensions=4, walk_length=8, num_walks=50)
    #
    # print("############")
    # # train node2vec model
    # n2w_model = node2vec_model.fit(window=7, min_count=1)
    # run_evaluate(data, n2w_model, alg="Node2Vec")


def preprocessing_graph(G: nx.Graph):
    node_list_1 = [u for u, _ in G.edges]
    node_list_2 = [v for _, v in G.edges]
    graph_df = pd.DataFrame({'node_1': node_list_1, 'node_2': node_list_2})
    all_unconnected_pairs = get_unconnected_pairs(G)
    node_1_unlinked = [i[0] for i in all_unconnected_pairs]
    node_2_unlinked = [i[1] for i in all_unconnected_pairs]
    data = pd.DataFrame({'node_1': node_1_unlinked,
                         'node_2': node_2_unlinked})
    # add target variable 'link'
    data['link'] = 0

    initial_node_count = len(G.nodes)
    graph_df_temp = graph_df.copy()
    # empty list to store removable links
    omissible_links_index = []
    for i in tqdm(graph_df.index.values):
        # remove a node pair and build a new graph
        G_temp = nx.from_pandas_edgelist(graph_df_temp.drop(index=i), "node_1", "node_2", create_using=nx.Graph())
        # check there is no spliting of graph and number of nodes is same
        if (nx.number_connected_components(G_temp) == 1) and (len(G_temp.nodes) == initial_node_count):
            omissible_links_index.append(i)
            graph_df_temp = graph_df_temp.drop(index=i)

    # create dataframe of removable edges
    removed_edge_graph_df = graph_df.loc[omissible_links_index]
    # add the target variable 'link'
    removed_edge_graph_df['link'] = 1

    data = data.append(removed_edge_graph_df[['node_1', 'node_2', 'link']], ignore_index=True)

    # drop removable edges
    graph_df_partial = graph_df.drop(index=removed_edge_graph_df.index.values)

    # build graph
    G_partial = nx.from_pandas_edgelist(graph_df_partial, "node_1", "node_2", create_using=nx.Graph())
    return data, G_partial


def plot_link_prediction_graph(G: nx.Graph, pred_edges: [], pred_acc=None):
    if pred_acc is None:
        pred_acc = []
    pos = nx.spring_layout(G, seed=6)  # positions for all nodes
    nx.draw_networkx_nodes(G, pos)
    # edges
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)

    for u, v in pred_edges:
        G.add_edge(u, v)
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=pred_edges,
        # width=1,
        # alpha=0.5,
        edge_color='r'
    )

    if pred_acc:
        edges_labels = {}
        for i in range(len(pred_edges)):
            edges_labels[(pred_edges[i][0], pred_edges[i][1])] = round(pred_acc[i], 2)
        # print(edges_labels)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edges_labels, font_color='red')

    labels = {}
    for u in G.nodes:
        labels[u] = str(u)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=16)

    plt.axis('off')
    plt.show()
