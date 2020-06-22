# -*- coding: utf-8 -*-
"""
Implementation of word2vec

Highlights:
1. Use loss function to learn word vector
2. Negative sampling (adding noise words) to combine: context_negative
#3. Hierarchical softmax (not used) 

Input: sentences
Output: word vector for every word (e.g. can be used for finding similarity)


@author: mayao
"""

import collections
import d2lzh as d2l
import math
from mxnet import autograd, gluon, nd
from mxnet.gluon import data as gdata, loss as gloss, nn
import random
import sys
import time
import zipfile

with zipfile.ZipFile('C:/Users/mayao/Desktop/d2l-zh/data/ptb.zip', 'r') as zin:
    zin.extractall('C:/Users/mayao/Desktop/d2l-zh/data/')

with open('C:/Users/mayao/Desktop/d2l-zh/data/ptb/ptb.train.txt', 'r') as f:
    lines = f.readlines()
    # st is sentence
    raw_dataset = [st.split() for st in lines]
    
counter = collections.Counter([tk for st in raw_dataset for tk in st])
counter = dict(filter(lambda x: x[1] >= 5, counter.items()))

#Idex to integer
idx_to_token = [tk for tk, _ in counter.items()]
token_to_idx = {tk: idx for idx, tk in enumerate(idx_to_token)}
dataset = [[token_to_idx[tk] for tk in st if tk in token_to_idx]
           for st in raw_dataset]
num_tokens = sum([len(st) for st in dataset])

# Second sampling with paremeter:1e-4
def discard(idx):
    return random.uniform(0, 1) < 1 - math.sqrt(
        1e-4 / counter[idx_to_token[idx]] * num_tokens)

subsampled_dataset = [[tk for tk in st if not discard(tk)] for st in dataset]

# Get centers and contexts with window 5
def get_centers_and_contexts(dataset, max_window_size):
    centers, contexts = [], []
    for st in dataset:
        if len(st) < 2:  # Two words at least for one sentence
            continue
        centers += st
        for center_i in range(len(st)):
            window_size = random.randint(1, max_window_size)
            indices = list(range(max(0, center_i - window_size),
                                 min(len(st), center_i + 1 + window_size)))
            indices.remove(center_i)  # Remove center word in background words
            contexts.append([st[idx] for idx in indices])
    return centers, contexts

all_centers, all_contexts = get_centers_and_contexts(subsampled_dataset, 5)

# Negative sampling
def get_negatives(all_contexts, sampling_weights, K):
    all_negatives, neg_candidates, i = [], [], 0
    population = list(range(len(sampling_weights)))
    for contexts in all_contexts:
        negatives = []
        while len(negatives) < len(contexts) * K:
            if i == len(neg_candidates):
                # Generate noise candidates a.c. sampling weights
                i, neg_candidates = 0, random.choices(
                    population, sampling_weights, k=int(1e5))
            neg, i = neg_candidates[i], i + 1
            # Take term which is not in contexts (Noise)
            if neg not in set(contexts):
                negatives.append(neg)
        all_negatives.append(negatives)
    return all_negatives

sampling_weights = [counter[w]**0.75 for w in idx_to_token]
all_negatives = get_negatives(all_contexts, sampling_weights, 5)

#Read batch size data, output mask (differ added 0 terms from others), label (differ context from others)
def batchify(data):
    max_len = max(len(c) + len(n) for _, c, n in data)
    centers, contexts_negatives, masks, labels = [], [], [], []
    for center, context, negative in data:
        cur_len = len(context) + len(negative)
        centers += [center]
        contexts_negatives += [context + negative + [0] * (max_len - cur_len)]
        masks += [[1] * cur_len + [0] * (max_len - cur_len)]
        labels += [[1] * len(context) + [0] * (max_len - len(context))]
    return (nd.array(centers).reshape((-1, 1)), nd.array(contexts_negatives),
            nd.array(masks), nd.array(labels))

#Get proper data 

batch_size = 512 #batch_size = 512
num_workers = 0 if sys.platform.startswith('win32') else 4
dataset = gdata.ArrayDataset(all_centers, all_contexts, all_negatives)
data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True,
                             batchify_fn=batchify, num_workers=num_workers)

# Settings 
def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    v = embed_v(center)
    u = embed_u(contexts_and_negatives)
    pred = nd.batch_dot(v, u.swapaxes(1, 2))
    return pred

loss = gloss.SigmoidBinaryCrossEntropyLoss()

embed_size = 100
net = nn.Sequential()
net.add(nn.Embedding(input_dim=len(idx_to_token), output_dim=embed_size),
        nn.Embedding(input_dim=len(idx_to_token), output_dim=embed_size))

# Training
def train(net, lr, num_epochs):
    ctx = d2l.try_gpu()
    net.initialize(ctx=ctx, force_reinit=True)
    trainer = gluon.Trainer(net.collect_params(), 'adam',
                            {'learning_rate': lr})
    for epoch in range(num_epochs):
        start, l_sum, n = time.time(), 0.0, 0
        for batch in data_iter:
            center, context_negative, mask, label = [
                data.as_in_context(ctx) for data in batch]
            with autograd.record():
                pred = skip_gram(center, context_negative, net[0], net[1])
                # Use mask to get rid of affecting loss functions
                l = (loss(pred.reshape(label.shape), label, mask) *
                     mask.shape[1] / mask.sum(axis=1))
            l.backward()
            trainer.step(batch_size)
            l_sum += l.sum().asscalar()
            n += l.size
        print('epoch %d, loss %.2f, time %.2fs'
              % (epoch + 1, l_sum / n, time.time() - start))
        

train(net, 0.005, 5)  #train(net, 0.005, 5)


#One little application
def get_similar_tokens(query_token, k, embed):
    W = embed.weight.data()
    x = W[token_to_idx[query_token]]
    # Add 1e-9 for numerical stability 
    cos = nd.dot(W, x) / (nd.sum(W * W, axis=1) * nd.sum(x * x) + 1e-9).sqrt()
    topk = nd.topk(cos, k=k+1, ret_typ='indices').asnumpy().astype('int32')
    for i in topk[1:]:  # remove input word
        print('cosine sim=%.3f: %s' % (cos[i].asscalar(), (idx_to_token[i])))

get_similar_tokens('apple', 3, net[0])

