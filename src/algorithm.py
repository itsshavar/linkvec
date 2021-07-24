import numpy as np
from gensim.models import Word2Vec


def str_list(walk):
    temp = []
    for i in walk:
        temp.append(str(i))
    return temp
def get_bias(probs):

    '''
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    '''
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        #print(K * prob)
        q[kk] = K * prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)
    return J, q


def alias_draw(J, q):
    '''
    Draw sample from a non-uniform discrete distribution using alias sampling.
    '''
    K = len(J)

    kk = int(np.floor(np.random.rand() * K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]



def linkvec(f, src,dest,strategy):
    vec1 = f.get_vector(str(src))
    vec2 = f.get_vector(str(dest))
    final_vector = []
    if strategy == 'max' : 
        for i,j in zip(vec1,vec2):
            if i >= j:
                final_vector.append(i)
            else:
                final_vector.append(j)

    else:
        for i,j in zip(vec1,vec2):
            if i  < j:
                final_vector.append(i)
            else:
                final_vector.append(j)
    return final_vector


def learn_embeddings(walks,dimension, window_size,workers,iter,output):
    '''
    Learn embeddings by optimizing the Skipgram objective using SGD.
    '''
    walks = [str_list(walk) for walk in walks]
    model = Word2Vec(
        walks,
        size=dimension,
        window=window_size,
        min_count=0,
        sg=1,
        workers=workers,
        iter=iter)
    model.wv.save_word2vec_format(output)

    return model.wv