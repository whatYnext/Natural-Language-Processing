# -*- coding: utf-8 -*-
"""
Use trained word vector to obtain similarity and analogy words from given word

@author: mayao
"""
from mxnet import nd
from mxnet.contrib import text

#Require time to download
glove_6b50d = text.embedding.create(
    'glove', pretrained_file_name='glove.6B.50d.txt')

#cos function value
def knn(W, x, k):
    # 1e-9 for numerical stability
    cos = nd.dot(W, x.reshape((-1,))) / (
        (nd.sum(W * W, axis=1) + 1e-9).sqrt() * nd.sum(x * x).sqrt())
    topk = nd.topk(cos, k=k, ret_typ='indices').asnumpy().astype('int32')
    return topk, [cos[i].asscalar() for i in topk]

def get_similar_tokens(query_token, k, embed):
    topk, cos = knn(embed.idx_to_vec,
                    embed.get_vecs_by_tokens([query_token]), k+1)
    for i, c in zip(topk[1:], cos[1:]):  # remove input word
        print('cosine sim=%.3f: %s' % (c, (embed.idx_to_token[i])))
        
get_similar_tokens('apple', 3, glove_6b50d)

def get_analogy(token_a, token_b, token_c, embed):
    vecs = embed.get_vecs_by_tokens([token_a, token_b, token_c])
    x = vecs[1] - vecs[0] + vecs[2]
    topk, cos = knn(embed.idx_to_vec, x, 1)
    return embed.idx_to_token[topk[0]]

get_analogy('bad', 'worst', 'big', glove_6b50d)