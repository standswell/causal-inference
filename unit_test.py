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
from poc import Causal_Graph


seed_test = False
var_test = False
real_test = True



if seed_test:
    # Unit Test 1: Test Random Seeds
    seeds = list(np.random.randint(low=1, high=1001, size=20))

    for seed in seeds:
        np.random.seed(seed)
        n = 10000
        X = np.random.normal(0, 1, size = n)
        Y = np.random.normal(0, 1, size = n)
        Z = 0.2*X + 1.5*Y + np.random.normal(0, 1, size = n)
        E = 0.5*Z + 0.8*X + np.random.normal(0, 1, size = n)
        data = pd.DataFrame({'X':X,'Y':Y,'Z':Z,'E':E})

        cg = Causal_Graph(data, experiment_name=f'seed_test_{seed}', experiment_folder='seed_tests')
        graph, edges , weights = cg.fit()

if var_test:
    # Unit Test 2: Variable Adjustment Tests

    # Sub-test: Increase the number of independent Variables
    seeds = list(np.random.randint(low=1, high=1001, size=20))

    for seed in seeds:
        np.random.seed(seed)
        n = 10000
        X = np.random.normal(0, 1, size = n)
        Y = np.random.normal(0, 1, size = n)

        F = np.random.normal(0, 1, size = n)
        K = np.random.normal(0, 1, size = n)
        C = np.random.normal(0, 1, size = n)

        Z = 0.2*X + 1.5*Y + 2.8* C + 0.01* K + np.random.normal(0, 1, size = n)
        E = 0.5*Z + 0.8*X + 4.8 * F + np.random.normal(0, 1, size = n)

        data = pd.DataFrame({'X':X,'Y':Y,'F':F,'K':K,'C':C,'Z':Z,'E':E})

        cg = Causal_Graph(data, experiment_name=f'more_independent_vars_{seed}', experiment_folder='variable_tests')
        graph, edges , weights = cg.fit()


if real_test:

    # College plan - discrete variable dataset (https://raw.githubusercontent.com/cmu-phil/example-causal-datasets/main/real/college-plans/data/college-plans.discrete.txt)
    # data = pd.read_csv('https://raw.githubusercontent.com/cmu-phil/example-causal-datasets/main/real/college-plans/data/college-plans.discrete.txt',sep='\t')

    # cg = Causal_Graph(data, experiment_name=f'College plans', experiment_folder='real_benchmarks')
    # graph, edges , weights = cg.fit()


    # Sachs - gene network data - continious (https://raw.githubusercontent.com/cmu-phil/example-causal-datasets/main/real/sachs/data/sachs.2005.continuous.txt)
    # data = pd.read_csv('https://raw.githubusercontent.com/cmu-phil/example-causal-datasets/main/real/sachs/data/sachs.2005.with.jittered.experimental.continuous.txt', sep='\t')

    # cg = Causal_Graph(data, experiment_name=f'Sachs2005', experiment_folder='real_benchmarks')
    # graph, edges , weights = cg.fit()

    #Superconductivity - continious dataset (https://raw.githubusercontent.com/cmu-phil/example-causal-datasets/main/real/superconductivity/data/superconductivity.continuous.txt)
    data = pd.read_csv('https://raw.githubusercontent.com/cmu-phil/example-causal-datasets/main/real/superconductivity/data/superconductivity.continuous.txt', sep='\t')

    cg = Causal_Graph(data, experiment_name=f'Superconductivity', experiment_folder='real_benchmarks')
    graph, edges , weights = cg.fit()

    #Red Wine quality - continious https://raw.githubusercontent.com/cmu-phil/example-causal-datasets/main/real/wine-quality/data/winequality-red.continuous.txt
    data = pd.read_csv('https://raw.githubusercontent.com/cmu-phil/example-causal-datasets/main/real/wine-quality/data/winequality-red.continuous.txt', sep='\t')

    cg = Causal_Graph(data, experiment_name=f'Red_Wine', experiment_folder='real_benchmarks')
    graph, edges , weights = cg.fit()

    # White Wine quality - continious https://raw.githubusercontent.com/cmu-phil/example-causal-datasets/main/real/wine-quality/data/winequality-red.continuous.txt
    data = pd.read_csv('https://raw.githubusercontent.com/cmu-phil/example-causal-datasets/main/real/wine-quality/data/winequality-white.continuous.txt', sep='\t')

    cg = Causal_Graph(data, experiment_name=f'White_Wine', experiment_folder='real_benchmarks')
    graph, edges , weights = cg.fit()












