# -*- coding: utf-8 -*-
"""
given a word and visualize near words
original source code is https://github.com/nishio/mycorpus/blob/master/vis.py
"""
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.font_manager
import numpy as np


class visWord2Vec:
    def __init__(self, filename='vectors.bin'):
        font = matplotlib.font_manager.FontProperties(fname='./ipag.ttc')
        FONT_SIZE = 20
        self.TEXT_KW = dict(fontsize=FONT_SIZE, fontweight='bold', fontproperties=font)

        print('loading')
        self.data = Word2Vec.load_word2vec_format(filename, binary=True)
        print('loaded')

    def plot(self, query, nbest=15):
        if ', ' not in query:
            words = [query] + list(map(lambda x: x[0], self.data.most_similar(query, topn=nbest)))
        else:
            words = query.split(', ')
            print(query)
        word_vectors = [self.data[x] for x in words]

        # do PCA
        X = np.stack(word_vectors)
        pca = PCA(n_components=2)
        pca.fit(X)
        print(pca.explained_variance_ratio_)
        X = pca.transform(X)
        xs = X[:, 0]
        ys = X[:, 1]

        # draw
        plt.figure(figsize=(12, 8))
        plt.scatter(xs, ys, marker='o')
        for i, w in enumerate(words):
            plt.annotate(
                w.decode('utf-8', 'ignore'),
                xy=(xs[i], ys[i]), xytext=(3, 3),
                textcoords='offset points', ha='left', va='top',
                **self.TEXT_KW)

        plt.show()
