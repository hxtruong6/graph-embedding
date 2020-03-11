import networkx as nx
from time import time
import matplotlib.pyplot as plt
import scipy.sparse as sp
from sklearn.preprocessing import normalize
import scipy.sparse.linalg as lg
import seaborn as sns
import numpy as np


def get_template_graph():
    return nx.karate_club_graph()

def get_template_parameter():
    return dict({
        "dimension": 2,
        
    })