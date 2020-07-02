from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from munkres import Munkres, print_matrix
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score, roc_auc_score
from scipy.optimize import linear_sum_assignment as linear
from sklearn import metrics
import lightgbm as lgbm
from node2vec import Node2Vec

from static_ge import StaticGE
from utils.classify import Classifier
from utils.visualize import read_node_label


def evaluate_classify_embeddings(embeddings, label_file=None):
    if label_file is None:
        raise ValueError("Must provide label_file name.")
    X, Y = read_node_label(filename=label_file)
    tr_frac = 0.8
    print("Training classifier using {:.2f}% nodes...".format(tr_frac * 100))
    clf = Classifier(embeddings=embeddings, clf=LogisticRegression())
    clf.split_train_evaluate(X, Y, tr_frac)


# https://github.com/bdy9527/SDCN
def cluster_acc(y_true, y_pred):
    y_true = y_true - np.min(y_true)

    l1 = list(set(y_true))
    numclass1 = len(l1)

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    ind = 0
    if numclass1 != numclass2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    if numclass1 != numclass2:
        print('error')
        return

    cost = np.zeros((numclass1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)

    # match two clustering results by Munkres algorithm
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)

    # get the match results
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        # correponding label in l2:
        c2 = l2[indexes[i][1]]

        # ai is the index with label==c2 in the pred_label list
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c

    acc = metrics.accuracy_score(y_true, new_predict)
    f1_macro = metrics.f1_score(y_true, new_predict, average='macro')
    precision_macro = metrics.precision_score(y_true, new_predict, average='macro')
    recall_macro = metrics.recall_score(y_true, new_predict, average='macro')
    f1_micro = metrics.f1_score(y_true, new_predict, average='micro')
    precision_micro = metrics.precision_score(y_true, new_predict, average='micro')
    recall_micro = metrics.recall_score(y_true, new_predict, average='micro')
    return acc, f1_macro


def cluster_evaluate(y_true, y_pred, alg=0):
    acc, f1 = cluster_acc(y_true, y_pred)
    nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
    ari = ari_score(y_true, y_pred)
    print(alg, ':acc {:.4f}'.format(acc), ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari),
          ', f1 {:.4f}'.format(f1))


def get_unconnected_pairs(G):
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


def run_evaluate(data, embedding, alg=None):
    if alg == "Node2Vec":
        x = [(embedding[str(i)] + embedding[str(j)]) for i, j in zip(data['node_1'], data['node_2'])]
    else:
        x = [(embedding[i] + embedding[j]) for i, j in zip(data['node_1'], data['node_2'])]

    xtrain, xtest, ytrain, ytest = train_test_split(np.array(x), data['link'],
                                                    test_size=0.3,
                                                    random_state=35)

    train_data = lgbm.Dataset(xtrain, ytrain)
    test_data = lgbm.Dataset(xtest, ytest)

    # define parameters
    parameters = {
        'objective': 'binary',
        'metric': 'auc',
        'is_unbalance': 'true',
        'feature_fraction': 0.5,
        'bagging_fraction': 0.5,
        'bagging_freq': 20,
        'num_threads': 2,
        'seed': 76
    }

    # train lightGBM model
    model = lgbm.train(parameters,
                       train_data,
                       valid_sets=test_data,
                       num_boost_round=1000,
                       early_stopping_rounds=20)

    # predictions = model.predict(xtest)
    # print(roc_auc_score(ytest, predictions))


def link_predict_evaluate(G):
    G = nx.Graph(G)
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

    # SDNE
    ge = StaticGE(G_data, embedding_dim=4, hidden_dims=[8])
    ge.train(epochs=40)
    embedding = ge.get_embedding()

    run_evaluate(data, embedding)

    # Node2vec
    node2vec_model = Node2Vec(G_data, dimensions=4, walk_length=8, num_walks=50)

    print("############")
    # train node2vec model
    n2w_model = node2vec_model.fit(window=7, min_count=1)
    run_evaluate(data, n2w_model, alg="Node2Vec")


if __name__ == "__main__":
    G = nx.karate_club_graph().to_undirected()
    link_predict_evaluate(G)
