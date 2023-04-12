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
from utils import convert_learned_graph_to_dictionary
import graphviz

import numpy as np

def generate_random_data(num_variables, dependence_structure):
    # Generate a covariance matrix with the specified dependence structure
    cov = np.ones((num_variables, num_variables)) * dependence_structure + np.eye(num_variables) * (1 - dependence_structure)

    print(cov)
    
    # Generate random data with the specified covariance matrix
    data = np.random.multivariate_normal(np.zeros(num_variables), cov, size=1000)
    
    return data

# Example usage
random_data = generate_random_data(num_variables=5, dependence_structure=0.2)



