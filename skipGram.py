from __future__ import division
import argparse
import pandas as pd
import re
import numpy as np
from scipy.special import expit
from sklearn.preprocessing import normalize
import collections
import pickle

__authors__ = ['Xinyu HU','Hugo VANDERPERRE','Anurag CHATTERJEE']
__emails__  = ['xinyu.hu@student-cs.fr','hugo.vanderperre@student-cs.fr','anurag.chatterjee@student-cs.fr']

def text2sentences(path):

    # some common punctuations to normalize the data
    punctuations = {'!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',',
                    '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', 
                    ']', '^', '_', '`', '{', '|', '}', '~',' '}
    
    sentences = []
    with open(path, encoding = 'utf-8') as f:
        for l in f:
            words = [''.join(ch for ch in word if ch not in punctuations) for word in l.lower().split()]
            if len(words) > 1:      
                sentences.append( words )
    return sentences


def loadPairs(path):
    data = pd.read_csv(path, delimiter='\t')
    pairs = zip(data['word1'], data['word2'], data['similarity'])
    return pairs

# define a sigmoid function for trainword backward propagation 
def sigmoid(x):
    if x > 5:
        return 1
    elif x < -5:
        return 0
    else:
        return 1/(1+np.exp(-x))

class SkipGram:
    def __init__(self, sentences=[], nEmbed=10, negativeRate=5, winSize=5, minCount=1):

        # store parameters as class parameters
        self.trainset = sentences
        self.nEmbed = nEmbed
        self.negativeRate = negativeRate
        self.minCount = minCount
        self.winSize = winSize

        # set learning rate for back propogation process
        self.lr = 0.002

        # constrcut a dictionary where words are keys and corresponding frequencies are values
        self.word_counts = collections.defaultdict(int)
        for row in sentences:
            for word in row:
                self.word_counts[word] += 1

        # filter words by mininal count
        self.vocab = {k: v for k, v in self.word_counts.items() if v > self.minCount}

        # reflect words as numbers
        self.w2id = dict(zip(self.vocab.keys(), np.arange(0, len(self.vocab))))
        id = self.w2id.values()

        # store the number of words in corpus
        self.count = len(self.vocab.keys())

        # initialize the embedding matrix W and C
        self.W = np.random.randn(self.count, nEmbed)
        self.C = np.random.randn(self.count, nEmbed)

        # initialize loss 
        self.loss = []
        self.trainWords = 0
        self.accLoss = 0.

        # compute the probability of each word
        self.prob = {}
        total_count = 0
        for w in self.w2id.keys():
            # penalize the counts of all the words by power of 3/4
            count = self.word_counts[w] ** (3 / 4)
            total_count += count
            self.prob[self.w2id[w]] = count
        # dictionary with keys are ids and values are probability(frequency)
        self.prob = {k: v / total_count for k, v in self.prob.items()}

    def sample(self, omit, negativeRate = 5):
        
        """
        samples negative words, ommitting those in set omit.
        Words that actually appear within the context window of the center word 
        and generate ids of words that are randomly drawn from a noise distribution

        Parameters
        ----------
        omit: tuple of {wIdx, ctxtId}
        wIdx is the index of center word, ctxtId is the index of context word 
        
        negetiveRate: a hyper-parameter that can be empircially tuned, the number of negative index
        which will be sampled

        Returns
        -------
        negativeIds: a list of index of negative words
        """

        # need to align with paper a bit
        prob_list = list(self.prob.values())
        id_list = list(self.prob.keys())

        # randomly choose k(negativerate) ids of words based on their probabilites
        negativeIds = np.random.choice(id_list, replace=False, size=self.negativeRate, p=prob_list)

        # replace ones which are already in corpus
        for i in range(negativeRate):
            if negativeIds[i] in omit:
                while negativeIds[i] in omit:
                    negativeIds[i] = np.random.choice(id_list, p=prob_list)
        return negativeIds

    def train(self, nb_epochs=10):
        eta = 0.25
        for epoch in range(nb_epochs):
            eta = 0.9 * eta
            for counter, sentence in enumerate(self.trainset):
                sentence = list(filter(lambda word: word in self.vocab, sentence))

                for wpos, word in enumerate(sentence):
                    wIdx = self.w2id[word]
                    winsize = np.random.randint(self.winSize) + 1
                    start = max(0, wpos - winsize)
                    end = min(wpos + winsize + 1, len(sentence))
                    for context_word in sentence[start:end]:
                        ctxtId = self.w2id[context_word]
                        if ctxtId == wIdx: continue
                        negativeIds = self.sample({wIdx, ctxtId})
                        self.trainWord(wIdx, ctxtId, negativeIds, eta)
                        self.trainWords += 1
                        self.accLoss += self.compute_loss(wIdx, ctxtId)
                if counter % 100 == 0:
                    # print(' > training %d of %d' % (counter, len(self.trainset)))
                    self.loss.append(self.accLoss / self.trainWords)
                    self.trainWords = 0
                    self.accLoss = 0.

    def trainWord(self, wordId, contextId, negativeIds, lr=0.002):

        W_update = 0
        W_update -= (sigmoid(np.dot(self.W[wordId,:], self.C[contextId, :])) - 1) * self.C[contextId, :]

        for negativeId in negativeIds:
            self.C[contextId, :] -= self.lr * sigmoid(np.dot(self.W[negativeId,:], self.C[contextId, :])) * self.W[wordId, :]
            W_update -= sigmoid(np.dot(self.W[negativeId,:], self.C[contextId, :])) * self.C[contextId, :]

        self.W[wordId, :] += self.lr * W_update

    def similarity(self, word1, word2, nEmbed=10):
        
        common_vect = +np.ones(self.nEmbed) * 10000
        if word1 not in self.vocab and word2 in self.vocab:
            id_word_2 = self.w2id[word2]
            w1 = common_vect
            w2 = self.W[id_word_2]
        elif word1 in self.vocab and word2 not in self.vocab:
            id_word_1 = self.w2id[word1]
            w1 = self.W[id_word_1]
            w2 = common_vect
        elif word1 not in self.vocab and word2 not in self.vocab:
            w1 = common_vect
            w2 = common_vect
        else:
            id_word_1 = self.w2id[word1]
            id_word_2 = self.w2id[word2]
            w1 = self.W[id_word_1]
            w2 = self.W[id_word_2]
        similarity = w1.dot(w2) / (np.linalg.norm(w1) * np.linalg.norm(w2))
        return similarity

    def compute_loss(self, wordId_1, wordId_2):

        # get the vectors of both of the words
        w1 = self.W[wordId_1]
        w2 = self.W[wordId_2]

        scalar = w1.dot(w2)
        similarity = 1 / (1 + np.exp(-scalar))
        return similarity
    
    def save(self, path):
      pickle.dump(self, open(path, 'wb'))

    
    @staticmethod
    def load(path):
      return pickle.load(open(path, 'rb'))

       
   

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--text', help='path containing training data', required=True)
    parser.add_argument('--model', help='path to store/read model (when training/testing)', required=True)
    parser.add_argument('--test', help='enters test mode', action='store_true')

    opts = parser.parse_args()

    if not opts.test:
        sentences = text2sentences(opts.text)
        sg = SkipGram(sentences,nEmbed=50, negativeRate=5, winSize=5, minCount=1)
        sg.train()
        sg.save(opts.model)

    else:
        pairs = loadPairs(opts.text)
        sg = SkipGram.load(opts.model)
        for a, b, true_score in pairs:
            print(sg.similarity(a, b))
  
            
