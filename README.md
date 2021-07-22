# linkvec

This repository provides a reference implementation of *linkvec* 


### Basic Usage

#### Example
To run *linkvec* on random network, execute the following command from the project home directory:<br/>
	``python src/main.py --input graph/random.edgelist --output emb/random.emb``

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

