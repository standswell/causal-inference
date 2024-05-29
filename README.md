# causal-inference

# My Project

## SVG Animation

Check out this cool animation [here](https://standswell.github.io/causal-inference/githubtest.html).


Welcome to our research project on the use of causal inference in analysing and modelling business problems. This project aims to explore the effectiveness of causal inference techniques in providing insights into various business problems and how it can help decision-makers make informed decisions.

Causal inference is a statistical approach that seeks to determine the causal relationship between variables. It is an essential tool for researchers who want to understand the underlying mechanisms that drive the outcomes of interest. In the context of business, causal inference can help in identifying the factors that influence business performance, predicting the impact of potential interventions, and evaluating the effectiveness of policies and strategies.

In this project, we will be using various causal inference methods, including causal discovery, causal effect estimation and quantifying interventions to analyze and model and visualize different business problems. We will explore how these techniques can be applied to real-world data sets, and how they can assist business decision-makers.

Our research project is hosted on Github, and we welcome contributions from anyone interested in this topic. We hope that this project will not only advance our understanding of causal inference in business but also provide a useful resource for anyone looking to apply these techniques to their own research projects.


# Get Started

Before initializing the repository it is required to have downloaded and install Graphviz, since the repository heavily utilizes its properties for visualization, you can download the latest version here,

```
https://graphviz.org/download/
```

Make sure that during the installation, you add Graphviz to the path variables for the users in your system that will be utilizing the repository.


Once you initialize your interpreter environment (we recommend python 3.10), make sure your working directory contains the requirements.txt file and install the required modules by executing the command below in your terminal,

```
pip install -r requirements.txt
```

Once the requirements are installed successfully you can demo the graph creation and estimation by executing the command below, 

```
python demo.py
```

This will run our proposed causal end-to-end pipeline through a list of demo datasets we have experimented with, that will be saved in your local environment under the directory demos.

Alternatively, you can test your own datasets by executing the following command,

```
python demo.py --data_path <data_path_to_test> --experiment_name <experiment_name>
```

By replacing <data_path_to_test> with the absolute path of your dataset and <experiment_name> with the name identification you want to trace your data in the demos directory. 
Keep in mind in the current implementation the data input need to be numeric and if not the columns containing non-numeric data will be removed from the data.


<svg width="400" height="180">
  <rect width="100%" height="100%" fill="red">
    <animate attributeName="fill" from="red" to="blue" dur="5s" repeatCount="indefinite"/>
  </rect>
</svg>





