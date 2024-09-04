import numpy as np
from Utils.twokenize import tokenize

def sum_vecs(sentence, model):
    """Returns the sum of the vectors of the tokens
    in the sentence if they are in the model"""
    sent = np.array(np.zeros((model.vector_size)))
    for w in sentence.split():
        try:
            sent += model[w]
        except:
            # TODO: implement a much better backoff strategy (Edit distance)
            pass
    return sent


def ave_vecs(sentence, model):
    sent = np.array(np.zeros((model.vector_size)))
    sent_length = len(sentence.split())
    for w in sentence.split():
        try:
            sent += model[w]
        except:
            # TODO: implement a much better backoff strategy (Edit distance)
            sent += model['the']
    return sent / sent_length


def idx_vecs(sentence, model):
    """Returns a list of vectors of the tokens
    in the sentence if they are in the model."""
    sent = []
    for w in sentence.split():
        try:
            sent.append(model[w])
        except:
            # TODO: implement a much better backoff strategy (Edit distance)
            sent.append(model['of'])
    return sent


def bow(sentence, model):
    """
    Bag of words representation
    """
    array = np.zeros(len(model))
    for w in sentence:
        try:
            array[model[w]] += 1
        except KeyError:
            pass
    return array

def simple_words(sentence, model):
    return sentence.split()

def words(sentence, model):
    return rem_mentions_urls(tokenize(sentence.lower()))

def rem_mentions_urls(tokens):
    final = []
    for t in tokens:
        if t.startswith('@'):
            final.append('at')
        elif t.startswith('http'):
            final.append('url')
        else:
            final.append(t)
    return final

def getMyData(fname, label, model, representation=sum_vecs, encoding='utf8'):
    data = []
    for sent in open(fname):
        data.append((representation(sent, model), label))
    return data

