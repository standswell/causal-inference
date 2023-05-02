from causallearn.search.ConstraintBased.PC import pc
import numpy as np
import graphviz
from causallearn.utils.GraphUtils import GraphUtils
from dowhy import gcm
import networkx as nx
import pydot
from pathlib import Path
import os

from utils import convert_learned_graph_to_dictionary

class Causal_Graph:

    def __init__(self, data, root_directory ,experiment_name, experiment_folder, alpha = 0.001, indep_test = 'fisherz', uc_priority = 4):
        self.data = data
        self.labels = list(self.data.columns)
        self.n = data.shape[0]

        self.root_directory = root_directory
        self.experiment_name = experiment_name
        self.experiment_folder = experiment_folder

        self.alpha = alpha
        self.indep_test = indep_test
        self.uc_priority = uc_priority

        self.path = os.path.join(Path.cwd(), self.root_directory, self.experiment_folder)

        if not os.path.exists(self.path):
            os.mkdir(self.path)

    def causal_discovery(self, data):
        ''' 
        Fits a graphical causal model given a graph
        Estimates the average causal effect of each edge in the graph given the list of the graph's edges
        '''

        cg = pc(np.array(data), alpha = self.alpha, indep_test = self.indep_test, uc_priority= self.uc_priority)


        pyd = GraphUtils.to_pydot(cg.G, labels = list(self.labels))

        pyd.write(f'{self.root_directory}/{self.experiment_folder}/learned_graph.dot')
        graph = pydot.graph_from_dot_file(f'{self.root_directory}/{self.experiment_folder}/learned_graph.dot')[0]
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

            g.render(f'demos/{self.experiment_folder}/graph_{self.experiment_name}', format='png')
        
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

    













