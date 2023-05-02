import pandas as pd
import argparse

from causal import Causal_Graph

import sys

parser = argparse.ArgumentParser()

parser.add_argument('--data_path', type=str, help='path to csv containing input data')
parser.add_argument('--experiment_name', type=str, help='name of the custom experiment')


args = parser.parse_args()

if args.data_path:

    # College plan - discrete variable dataset (https://raw.githubusercontent.com/cmu-phil/example-causal-datasets/main/real/college-plans/data/college-plans.discrete.txt)
    data = pd.read_csv(args.data_path)

    data = data.dropna()
    data = data.select_dtypes(include=['int', 'float'])

    cg = Causal_Graph(data, root_directory='demos', experiment_name='Custom Experiment', experiment_folder=str(args.experiment_name))
    graph, edges, weights = cg.fit()

else:

    # College plan - discrete variable dataset (https://raw.githubusercontent.com/cmu-phil/example-causal-datasets/main/real/college-plans/data/college-plans.discrete.txt)
    data = pd.read_csv('https://raw.githubusercontent.com/cmu-phil/example-causal-datasets/main/real/college-plans/data/college-plans.discrete.txt',sep='\t')

    cg = Causal_Graph(data, root_directory='demos',experiment_name='College plans discrete', experiment_folder='College plans')
    graph, edges , weights = cg.fit()





