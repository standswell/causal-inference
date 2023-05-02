import pandas as pd
import numpy as np

from causal import Causal_Graph

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

    cg = Causal_Graph(data, root_directory='tests',experiment_name=f'seed_test_{seed}', experiment_folder='seed_tests')
    graph, edges , weights = cg.fit()

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

    cg = Causal_Graph(data, root_directory='tests',experiment_name=f'more_independent_vars_{seed}', experiment_folder='variable_tests')
    graph, edges , weights = cg.fit()












