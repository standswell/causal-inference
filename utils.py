import pandas as pd
from causallearn.search.ConstraintBased.PC import pc
import sys
import numpy as np
import pdb
from sklearn.preprocessing import LabelEncoder
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
import graphviz
import dowhy 
from causallearn.utils.GraphUtils import GraphUtils
from dowhy import CausalModel
import dowhy.datasets
from dowhy import gcm
import networkx as nx


def estimate_graph_average_causal_effects(model, model_fit, edges, n = 10000):

    weights = list()
    
    for edge in edges:
        weight = model_fit.average_causal_effect(model,
                         edge[1],
                         interventions_alternative={edge[0]: lambda x: 1},
                         interventions_reference={edge[0]: lambda x: 0},
                         num_samples_to_draw=n)
        weight = round(weight, 2)
        
        weights.append(weight)

    return weights

def convert_learned_graph_to_dictionary(graph, labels):
    edges = []
    for edge in graph.get_edges():
        source = edge.get_source()
        destination = edge.get_destination()
        edges.append((source, destination))

    for i, (source, destination) in enumerate(edges):
        edges[i] = (labels[int(source)], labels[int(destination)])
    
    return edges



def estimate_graph_weights(model, graph):

    weights = list()

    for edge in edges:
        ace = gcm.average_causal_effect(model,
                            edge[0].name,
                            interventions_alternative={edge[1].name: lambda x: 1},
                            interventions_reference={edge[1].name: lambda x: 0},
                            num_samples_to_draw=1000)
        weights.append(ace)

    return weights

def add_weights_to_graph_edges(weights, graph):




    return weighted_graph

