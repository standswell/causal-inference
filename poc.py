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
from dowhy.gcm.uncertainty import estimate_variance
import networkx as nx
import matplotlib.pyplot as plt
import pydot
from utils import convert_learned_graph_to_dictionary, estimate_graph_average_causal_effects
import pygraphviz


class Causal_Graph:

    def __init__(self, data, experiment_name, experiment_folder, alpha = 0.001, indep_test = 'fisherz', uc_priority = 4):
        self.data = data
        self.labels = list(self.data.columns)
        self.n = data.shape[0]
        self.experiment_name = experiment_name
        self.experiment_folder = experiment_folder
        self.alpha = alpha
        self.indep_test = indep_test
        self.uc_priority = uc_priority

    def causal_discovery(self, data):
        ''' 
        Fits a graphical causal model given a graph
        Estimates the average causal effect of each edge in the graph given the list of the graph's edges
        '''

        cg = pc(np.array(data), alpha = self.alpha, indep_test = self.indep_test, uc_priority= self.uc_priority)


        pyd = GraphUtils.to_pydot(cg.G, labels = list(self.labels))
        pyd.write_dot(f'tests/{self.experiment_folder}/learned_graph.dot')
        graph = pydot.graph_from_dot_file(f'tests/{self.experiment_folder}/learned_graph.dot')[0]
        edges = convert_learned_graph_to_dictionary(graph, self.labels)
        G = nx.DiGraph(edges)
        self.render_graph(edges, weights = False)

        return G, edges
    
    def causal_inference(self, graph, edges):
        ''' 
        Fits a graphical causal model given a graph
        Estimates the average causal effect of each edge in the graph given the list of the graph's edges
        '''

        causal_model = gcm.StructuralCausalModel(graph)
        gcm.auto.assign_causal_mechanisms(causal_model, self.data)
        gcm.fit(causal_model, self.data)

        weights = list()
        for edge in edges:
            weight = gcm.average_causal_effect(causal_model,
                                edge[1],
                                interventions_alternative={edge[0]: lambda x: 1},
                                interventions_reference={edge[0]: lambda x: 0},
                                num_samples_to_draw=self.n)
            weight = round(weight, 2)
            
            weights.append(weight)
        
        return weights

    def render_graph(self, edges, weights):

        if weights:
            g = graphviz.Digraph()
            for node in self.labels:
                g.node(node)

            for i in range(len(edges)):
                g.edge(edges[i][0], edges[i][1], label=str(weights[i]))

            g.render(f'tests/{self.experiment_folder}/graph_{self.experiment_name}', format='png')
        
        else:
            g = graphviz.Digraph()
            for node in self.labels:
                g.node(node)

            for i in range(len(edges)):
                g.edge(edges[i][0], edges[i][1])

            g.render(f'tests/{self.experiment_folder}/unweighted_graph_{self.experiment_name}', format='png')
    
    def fit(self):

        graph, edges = self.causal_discovery(self.data)
        weights = self.causal_inference(graph, edges)
        self.render_graph(edges, weights)

        return graph, edges , weights

    













