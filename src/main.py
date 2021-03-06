import argparse
import numpy as np
import networkx as nx
import linkvectorizer
import algorithm
from gensim.models import Word2Vec


def parse_args():
    '''
    Parses the linkvec arguments.
    '''
    parser = argparse.ArgumentParser(description="Run linkvec.")

    parser.add_argument('--input', nargs='?', default='graph/random.edgelist',
                        help='Input graph path')

    parser.add_argument('--output', nargs='?', default='emb/random.emb',
                        help='Embeddings path')

    parser.add_argument('--dimensions', type=int, default=10,
                        help='Number of dimensions. Default is 128.')

    parser.add_argument('--walk-length', type=int, default=3,
                        help='Length of walk per source. Default is 80.')

    parser.add_argument('--num-walks', type=int, default=1,
                        help='Number of walks per source. Default is 10.')

    parser.add_argument('--window-size', type=int, default=10,
                        help='Context size for optimization. Default is 10.')

    parser.add_argument('--iter', default=1, type=int,
                        help='Number of epochs in SGD')

    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers. Default is 8.')

    parser.add_argument('--p', type=float, default=1,
                        help='Return hyperparameter. Default is 1.')

    parser.add_argument('--q', type=float, default=1,
                        help='Inout hyperparameter. Default is 1.')

    parser.add_argument(
        '--weighted',
        dest='weighted',
        action='store_true',
        help='Boolean specifying (un)weighted. Default is unweighted.')
    parser.add_argument(
        '--unweighted',
        dest='unweighted',
        action='store_false')
    parser.set_defaults(weighted=False)

    parser.add_argument('--directed', dest='directed', action='store_true',
                        help='Graph is (un)directed. Default is undirected.')
    parser.add_argument(
        '--undirected',
        dest='undirected',
        action='store_false')
    parser.set_defaults(directed=False)

    return parser.parse_args()


def read_graph(args):
    '''
    Reads the input network in networkx.
    '''
    if args.weighted:
        G = nx.read_edgelist(args.input, nodetype=int, data=(
            ('weight', float),), create_using=nx.DiGraph())
    else:
        G = nx.read_edgelist(
            args.input,
            nodetype=int,
            create_using=nx.DiGraph())
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1

    if not args.directed:
        G = G.to_undirected()

    return G


def str_list(walk):
    temp = []
    for i in walk:
        temp.append(str(i))
    return temp


def learn_embeddings(walks):
    '''
    Learn embeddings by optimizing the Skipgram objective using SGD.
    '''
    walks = [str_list(walk) for walk in walks]
    model = Word2Vec(
        walks,
        size=args.dimensions,
        window=args.window_size,
        min_count=0,
        sg=1,
        workers=args.workers,
        iter=args.iter)
    model.wv.save_word2vec_format(args.output)

    return model.wv


def main(args):
    '''
    Pipeline for representational learning for all nodes in a graph.
    '''
    nx_G = read_graph(args)
    G = linkvectorizer.LinkVectorizer(nx_G, args.directed, args.p, args.q)
    G.preprocess_modified_weights()
    walks = G.generate_random_walks_with_bias(args.num_walks, args.walk_length)
    embeddings = learn_embeddings(walks)
     # Lets check the embedding for first two nodes.
    nodes = list(nx_G.nodes)
    src = nodes[0]
    dest = nodes[1]
    vec = algorithm.linkvec(embeddings,src,dest, strategy='max')
    print('Generated Vector is = >', vec)


if __name__ == "__main__":
    args = parse_args()
    main(args)
