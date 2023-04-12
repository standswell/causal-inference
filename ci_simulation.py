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
from causallearn.utils.DAG2CPDAG import dag2cpdag
from causallearn.search.ConstraintBased.FCI import fci


# data = dowhy.datasets.linear_dataset(
#     beta=10,
#     num_common_causes=5,
#     num_instruments=2,
#     num_samples=10000)

# print(data["df"])
# print(data["treatment_name"])
# print(data["outcome_name"])
#sys.exit()


n = 10000
X = np.random.normal(0, 1, size = n)
Y = np.random.normal(0, 1, size = n)
Z = 0.2*X + 1.5*Y + np.random.normal(0, 1, size = n)
E = 0.5*Z + 0.8*X + np.random.normal(0, 1, size = n)

data = pd.DataFrame({'X':X,'Y':Y,'Z':Z,'E':E})
labels = data.columns
data_arr = np.array(data)


cg = pc(data_arr)
#G, edges = fci(data_arr)
#print(cg)

pyd = GraphUtils.to_pydot(cg.G, labels = list(labels))

pyd.write_dot('learned_graph.dot')

#print(pyd)
#pyd.write_png('simulated_data_graph_pc.png')
sys.exit()

print(pyd.get_nodes()[0].get_label())

sys.exit()


#print(pyd.get_edges())


# # remove duplicate nodes
# added_nodes = set()
# for i, node in enumerate(pyd.get_nodes()):
#     if node.get_name() not in added_nodes:
#         added_nodes.add(node.get_name())
#     else:
#         pyd.del_node(i)

# print(added_nodes)
# print(pyd)

# delete every second node
nodes = pyd.get_nodes()
for i in range(0, len(nodes), 2):
    print(nodes[i].get_label())

print(pyd)


sys.exit()


# convert the pydot graph to a NetworkX graph
nx_graph = nx.DiGraph()
for pydot_node in pyd.get_nodes():
    nx_graph.add_node(pydot_node.get_name())
for pydot_edge in pyd.get_edges():
    nx_graph.add_edge(pydot_edge.get_source(), pydot_edge.get_destination())

#print(nx_graph.nodes)
# print(nx_graph.edges)
#print(nx_graph)

nx.write_gml(nx_graph, "graph.gml")
gml_string = nx.write_gml(nx_graph,"graph.gml")

#sys.exit()
# causal_model = gcm.StructuralCausalModel(pyd)
# gcm.auto.assign_causal_mechanisms(causal_model, data)

# model = CausalModel(
#     data=data,
#     treatment=["X"],
#     outcome="E",
#     graph = gml_string)

# # # Step 2: Identify causal effect and return target estimands
# identified_estimand = model.identify_effect()
# print(identified_estimand)

causal_model = gcm.StructuralCausalModel(gml_string)
gcm.auto.assign_causal_mechanisms(causal_model, data)
gcm.fit(causal_model, data)
