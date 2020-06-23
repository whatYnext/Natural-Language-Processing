# -*- coding: utf-8 -*-
"""
Implementation of textcnn

Highlights:
    1. Usage of 1 dimensional convolution (different length of kernels)
    2. Multi-channel's time max pooling
    3. Usage of drop out getting rid of overfit

@author: mayao
"""

import d2lzh as d2l
from mxnet import gluon, init, nd
from mxnet.contrib import text
from mxnet.gluon import data as gdata, loss as gloss, nn
import os
import random

#Get data
batch_size = 64
def read_imdb(folder='train'):  
    data = []
    for label in ['pos', 'neg']:
        folder_name = os.path.join('C:/Users/mayao/Desktop/d2l-zh/data/aclImdb/', folder, label)
        for file in os.listdir(folder_name):
            with open(os.path.join(folder_name, file), 'rb') as f:
                review = f.read().decode('utf-8').replace('\n', '').lower()
                data.append([review, 1 if label == 'pos' else 0])
    random.shuffle(data)
    return data

train_data, test_data = read_imdb('train'), read_imdb('test')

vocab = d2l.get_vocab_imdb(train_data)
train_iter = gdata.DataLoader(gdata.ArrayDataset(
    *d2l.preprocess_imdb(train_data, vocab)), batch_size, shuffle=True)
test_iter = gdata.DataLoader(gdata.ArrayDataset(
    *d2l.preprocess_imdb(test_data, vocab)), batch_size)

#Define textcnn
class TextCNN(nn.Block):
    def __init__(self, vocab, embed_size, kernel_sizes, num_channels,
                 **kwargs):
        super(TextCNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(len(vocab), embed_size)
        # Embedding without training
        self.constant_embedding = nn.Embedding(len(vocab), embed_size)
        self.dropout = nn.Dropout(0.5)
        self.decoder = nn.Dense(2)
        #One time maxpooling
        self.pool = nn.GlobalMaxPool1D()
        self.convs = nn.Sequential()  # create multi 1 dimensional cnn
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.add(nn.Conv1D(c, k, activation='relu'))

    def forward(self, inputs):
        # (batchsize, len(vocab), emb_size) contaced by emb_size
        embeddings = nd.concat(
            self.embedding(inputs), self.constant_embedding(inputs), dim=2)
        # Output:(batchsize,  emb_size, len(vocab))
        embeddings = embeddings.transpose((0, 2, 1))
        # Output:(batchsize, outputchannels,1) and then connect in the second dimension
        encoding = nd.concat(*[nd.flatten(
            self.pool(conv(embeddings))) for conv in self.convs], dim=1)
        # Apply dropout and then fully connected layer
        outputs = self.decoder(self.dropout(encoding))
        return outputs
#Settings   
embed_size, kernel_sizes, nums_channels = 100, [3, 4, 5], [100, 100, 100]
ctx = d2l.try_all_gpus()
net = TextCNN(vocab, embed_size, kernel_sizes, nums_channels)
net.initialize(init.Xavier(), ctx=ctx)

#Training
glove_embedding = text.embedding.create(
    'glove', pretrained_file_name='glove.6B.100d.txt', vocabulary=vocab)
net.embedding.weight.set_data(glove_embedding.idx_to_vec)
net.constant_embedding.weight.set_data(glove_embedding.idx_to_vec)
net.constant_embedding.collect_params().setattr('grad_req', 'null')

lr, num_epochs = 0.001, 5
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
loss = gloss.SoftmaxCrossEntropyLoss()
d2l.train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs)

#Prediction
d2l.predict_sentiment(net, vocab, ['this', 'movie', 'is', 'so', 'great'])
d2l.predict_sentiment(net, vocab, ['this', 'movie', 'is', 'so', 'bad'])