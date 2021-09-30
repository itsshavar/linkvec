# linkvec

This repository provides a reference implementation of *linkvec* 
This repository contains following files:
 1. `./emb/embedding.emb` : Contains the embedding for the given graph
 2. `./graph/fb-food-pages.edgelist` : Contains the edgelist for facebook food page [dataset](http://networkrepository.com/fb-pages-food.php)
 3. `./src/main.py` : Contains driver program for embedding generation
 4. `./src/linkprediciton.py` : Contains driver program for link predicition problem
 5. `./src/algorithm.py` : Contains methods for edge embedding.
 6. `./src/linkvectorizer.py` : Contains implementation of link vectorizer object.

## Embedding Generation

### Basic Usage

#### Example
To run *linkvec* on random network, execute the following command from the project home directory:<br/>
	``python src/main.py --input graph/fb-food-pages.edgelist --output emb/random.emb``

#### Options
You can check out the other options available to use with *linkvec* using:<br/>
	``python src/main.py --help``

#### Input
The supported input format is an edgelist:

	node1_id_int node2_id_int <weight_float, optional>
		
The graph is assumed to be undirected and unweighted by default. These options can be changed by setting the appropriate flags.

#### Output
The intermediatery output file has *n+1* lines for a graph with *n* vertices. 
The first line has the following format:

	num_of_nodes dim_of_representation

The next *n* lines are as follows:
	
	node_id dim1 dim2 ... dimd
the final edge embedding would be
<Edge : e1 ><dim1 ,dim2,  ... dimd>

## Link Prediction

To predict links one can use ```src/linkprediction.py``` 

The given implementation is for computing the accuracy. Rest of the metrics can be computed accordingly.

Citation :

@article{TRIPATHI2021,
title = {Network embedding based link prediction in dynamic networks},
journal = {Future Generation Computer Systems},
year = {2021},
issn = {0167-739X},
doi = {https://doi.org/10.1016/j.future.2021.09.024},
url = {https://www.sciencedirect.com/science/article/pii/S0167739X21003708},
author = {Shashi Prakash Tripathi and Rahul Kumar Yadav and Abhay Kumar Rai},
keywords = {Network embedding, Link prediction, Similarity measures, Network features, Feature learning, Biased random walk, Dynamic networks},
}
