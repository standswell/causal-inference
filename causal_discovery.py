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

# DOCS: https://causal-learn.readthedocs.io/_/downloads/en/latest/pdf/


#pdb.set_trace()
# default parameters
data = pd.read_csv('https://raw.githubusercontent.com/Sid-darthvader/DoWhy-The-Causal-Story-Behind-Hotel-Booking-Cancellations/master/hotel_bookings.csv')
# #print(data)
data = data.dropna()
# #data = data._get_numeric_data()
# #
# #print(data)
labels = data.columns
print(labels)

cols = data.columns
for col in cols:
    if is_numeric_dtype(data[col]):
        continue
    else:
        data[col] = data[col].astype('category').cat.codes

data = np.array(data)
cg = pc(data)

# visualization using pydot
#cg.draw_pydot_graph(labels = list(labels))

# or save the graph
from causallearn.utils.GraphUtils import GraphUtils

pyd = GraphUtils.to_pydot(cg.G, labels = list(labels))
#gvz=graphviz.Source(pyd)
#gvz.render('my_graph', view=False, format='pdf')
pyd.write_png('simple_test.png')