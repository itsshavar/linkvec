import pandas as pd
import numpy as np
import random
import networkx as nx
from tqdm.auto import tqdm
import re
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import  accuracy_score

import pickle
import timeit
from time import time
import linkvectorizer
from algorithm import linkvec,learn_embeddings

with open("data/fb-food-pages/fb-pages-food.nodes") as f:
    fb_nodes = f.read().splitlines() 

# load edges (or links)
with open("data/fb-food-pages/fb-pages-food.edges") as f:
    fb_links = f.read().splitlines() 

print("Node and Edge Length : ",len(fb_nodes), len(fb_links))

node_list_1 = []
node_list_2 = []

for i in tqdm(fb_links):
  node_list_1.append(i.split(',')[0])
  node_list_2.append(i.split(',')[1])

fb_df = pd.DataFrame({'node_1': node_list_1, 'node_2': node_list_2})

G = nx.from_pandas_edgelist(fb_df, "node_1", "node_2", create_using=nx.Graph())

print("Density of the Network",nx.density(G))

print(nx.number_connected_components(G))

node_list = node_list_1 + node_list_2

# remove duplicate items from the list
node_list = list(dict.fromkeys(node_list))

# build adjacency matrix
adj_G = nx.to_numpy_matrix(G, nodelist = node_list)

path = dict(nx.all_pairs_shortest_path(G))


all_unconnected_pairs = []

# traverse adjacency matrix
offset = 0
for i in tqdm(range(adj_G.shape[0])):
  for j in range(offset,adj_G.shape[1]):
      if i != j and adj_G[i,j] == 0:
        try:
          if len(path[str(i)][str(j)]) - 1 <=2:
            all_unconnected_pairs.append([node_list[i],node_list[j]])
        except:
          pass
  offset = offset + 1

print("Length of unconnected pair :",len(all_unconnected_pairs))

node_1_unlinked = [i[0] for i in all_unconnected_pairs]
node_2_unlinked = [i[1] for i in all_unconnected_pairs]

data = pd.DataFrame({'node_1':node_1_unlinked, 
                     'node_2':node_2_unlinked})

# add target variable 'link'
data['link'] = 0

initial_node_count = len(G.nodes)

fb_df_temp = fb_df.copy()

# empty list to store removable links
omissible_links_index = []

for i in tqdm(fb_df.index.values):
  
  # remove a node pair and build a new graph
  G_temp = nx.from_pandas_edgelist(fb_df_temp.drop(index = i), "node_1", "node_2", create_using=nx.Graph())
  
  # check there is no spliting of graph and number of nodes is same
  if (nx.number_connected_components(G_temp) == 1) and (len(G_temp.nodes) == initial_node_count):
    omissible_links_index.append(i)
    fb_df_temp = fb_df_temp.drop(index = i)

fb_df_ghost = fb_df.loc[omissible_links_index]

# add the target variable 'link'
fb_df_ghost['link'] = 1

data = data.append(fb_df_ghost[['node_1', 'node_2', 'link']], ignore_index=True)

data['link'].value_counts()

# drop removable edges
fb_df_partial = fb_df.drop(index=fb_df_ghost.index.values)

# build graph
G_data = nx.from_pandas_edgelist(fb_df_partial, "node_1", "node_2", create_using=nx.Graph())

for edge in G_data.edges():
            G_data[edge[0]][edge[1]]['weight'] = 1
G_data = G_data.to_undirected()
# Generate walks
G = linkvectorizer.LinkVectorizer(G_data,False, 1,1)
G.preprocess_modified_weights()
walks = G.generate_random_walks_with_bias(50, 16)
embeddings = learn_embeddings(walks,100,7,4,10,'emb/embedding.emb')

x = [linkvec(embeddings,i,j, strategy='max') for i,j in zip(data['node_1'], data['node_2'])]

xtrain, xtest, ytrain, ytest = train_test_split(np.array(x), data['link'], 
                                                test_size = 0.35, 
                                                random_state = 35)

import lightgbm as lgbm

train_data = lgbm.Dataset(xtrain, ytrain)
test_data = lgbm.Dataset(xtest, ytest)

# define parameters
parameters = {
    'objective': 'binary',
    'metric': 'auc',
    'is_unbalance': 'true',
    'feature_fraction': 0.5,
    'bagging_fraction': 0.5,
    'bagging_freq': 20,
    'num_threads' : 2,
    'seed' : 76
}

# train lightGBM model
#start = time()
model = lgbm.train(parameters, train_data,\
                         valid_sets=test_data,\
                         num_boost_round=1000,\
                         early_stopping_rounds=20)
# end = time()
# print(end-start)


ypred = model.predict(xtest, num_iteration=model.best_iteration)
y_pred = np.where(ypred > 0.8, 1, 0) 
print("Acc","=>",accuracy_score(ytest, y_pred))