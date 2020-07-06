# -*- coding: utf-8 -*-
"""
Use pre-trained word vector and a birnn(lstm) to give sentiment analysis on IMDB dataset

Highlights:
    1. Trained word vectors
    2. Bi-RNN (two hidden layers with every layer 100 hidden numbers)
    3. Use the first and last time output in the last RNN layer.

Output:
    1. Trained bi-rnn network with ability to judge positive or negative comments
@author: mayao
"""

import collections
import d2lzh as d2l
from mxnet import gluon, init, nd
from mxnet.contrib import text
from mxnet.gluon import data as gdata, loss as gloss, nn, rnn, utils as gutils
import os
import random
import tarfile

#Download imdb
# def download_imdb(data_dir='C:/Users/mayao/Desktop/d2l-zh/data'):
#     url = ('http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz')
#     sha1 = '01ada507287d82875905620988597833ad4e0903'
#     fname = gutils.download(url, data_dir, sha1_hash=sha1)
#     with tarfile.open(fname, 'r') as f:
#         f.extractall(data_dir)

# download_imdb()

#Read imdb
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

#Get data with different words being separated by " "
def get_tokenized_imdb(data):  
    def tokenizer(text):
        return [tok.lower() for tok in text.split(' ')]
    return [tokenizer(review) for review, _ in data]

#Create vocabbulary (with index)
def get_vocab_imdb(data):  
    tokenized_data = get_tokenized_imdb(data)
    counter = collections.Counter([tk for st in tokenized_data for tk in st])
    return text.vocab.Vocabulary(counter, min_freq=5,
                                 reserved_tokens=['<pad>'])

vocab = get_vocab_imdb(train_data)

def preprocess_imdb(data, vocab):  
    max_l = 500  # add <0> to have 500 length

    def pad(x):
        return x[:max_l] if len(x) > max_l else x + [
            vocab.token_to_idx['<pad>']] * (max_l - len(x))

    tokenized_data = get_tokenized_imdb(data)
    features = nd.array([pad(vocab.to_indices(x)) for x in tokenized_data])
    labels = nd.array([score for _, score in data])
    return features, labels

batch_size = 64
train_set = gdata.ArrayDataset(*preprocess_imdb(train_data, vocab))
test_set = gdata.ArrayDataset(*preprocess_imdb(test_data, vocab))
train_iter = gdata.DataLoader(train_set, batch_size, shuffle=True)
test_iter = gdata.DataLoader(test_set, batch_size)


class BiRNN(nn.Block):
    def __init__(self, vocab, embed_size, num_hiddens, num_layers, **kwargs):
        super(BiRNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(len(vocab), embed_size)
        # bidirectional=True
        self.encoder = rnn.LSTM(num_hiddens, num_layers=num_layers,
                                bidirectional=True, input_size=embed_size)
        self.decoder = nn.Dense(2)

    def forward(self, inputs): #Here Input's len(comment)=500
        # Inputs is (batchsize, len(comment)). Output is (len(comment), batchsize, embed_size)
        embeddings = self.embedding(inputs.T)
        # Outputs is (len(comment), batchsize, 2 *hidde_num) for the last layer
        outputs = self.encoder(embeddings)
        # Contact first and last time output which is (batchsize, 4*hidden_num)
        encoding = nd.concat(outputs[0], outputs[-1])
        outs = self.decoder(encoding)
        return outs

embed_size, num_hiddens, num_layers, ctx = 100, 100, 2, d2l.try_all_gpus()
net = BiRNN(vocab, embed_size, num_hiddens, num_layers)
net.initialize(init.Xavier(), ctx=ctx)

glove_embedding = text.embedding.create(
    'glove', pretrained_file_name='glove.6B.100d.txt', vocabulary=vocab)

#No need upgrading word vectors (it is trained on a larger data set)
net.embedding.weight.set_data(glove_embedding.idx_to_vec)
net.embedding.collect_params().setattr('grad_req', 'null')

lr, num_epochs = 0.01, 5 # 0.01, 5
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
loss = gloss.SoftmaxCrossEntropyLoss()
d2l.train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs)

#Prediction
def predict_sentiment(net, vocab, sentence):
    sentence = nd.array(vocab.to_indices(sentence), ctx=d2l.try_gpu())
    label = nd.argmax(net(sentence.reshape((1, -1))), axis=1)
    return 'positive' if label.asscalar() == 1 else 'negative'

#predict_sentiment(net, vocab, ['this', 'movie', 'is', 'so', 'great'])
#predict_sentiment(net, vocab, ['this', 'movie', 'is', 'so', 'bad'])
