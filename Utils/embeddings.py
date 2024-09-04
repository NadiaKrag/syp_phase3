import os, re
import numpy as np
from gensim.models import Word2Vec
from Utils.twokenize import tokenize


def words(sentence):
    return rem_mentions_urls(tokenize(sentence))


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


def generate_word2vec_embeddings(sentences, name, sg=1, size=300, sample=10**-4, window=5, negative=15, workers=4):
    model = Word2Vec(sentences, sg=sg, size=size, sample=sample, window=window, workers=workers)
    # Generate and concatenate oov embedding
    boundary = np.sqrt(3*model.wv.vectors.var(axis=0))
    oov = np.random.uniform(-boundary, boundary)
    model.wv.add(["<OOV>"], [oov])
    model.wv.save_word2vec_format("embeddings/{}_embeddings.txt".format(name))


def open_unlabeled_amazon(DIR, rep=words):
    data = open(os.path.join(DIR,'unlabeled.review'), encoding='latin-1').read()
    split = data.split('<review>')[1:]
    rm_tags = [get_between(l, '<review_text>\n', '\n</review_text>') for l in split]
    clean = [clean_str(s) for s in rm_tags]
    tokens = [rep(sent) for sent in clean]
    return np.array(tokens)


def open_twitter(DIR, binary=True, rep=words):
    train = []
    for line in open(os.path.join(DIR, 'test.tsv')):
        try:
            idx, sidx, label, tweet = line.split('\t')
        except ValueError:
            idx, label, tweet = line.split('\t', 2)
        train.append(tweet)
    tokens = [rep(sent) for sent in train]
    # flat = [i for lst in tokens for i in lst]
    # print(len(flat))
    return np.array(tokens)


def get_between(x, l, r):
    mid = x.split(l)[1]
    return mid.split(r)[0]


def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"'", " ' ", string)
    string = re.sub(r'"', ' " ', string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if TREC else string.strip().lower()


workers = 4
print('generating books embeddings')
books = open_unlabeled_amazon("datasets/amazon-multi-domain/books/")
generate_word2vec_embeddings(books, "books")

print('tokenizing dvd reviews')
dvd = open_unlabeled_amazon("datasets/amazon-multi-domain/dvd/")
print('generating dvd embeddings')
generate_word2vec_embeddings(dvd,"dvd", workers=workers)
dvd = None

print('tokenizing electronics reviews')
electronics = open_unlabeled_amazon("datasets/amazon-multi-domain/electronics/")
print('generating electronics embeddings')
generate_word2vec_embeddings(electronics,"electronics", workers=workers)
electronics = None

print('tokenizing kitchen reviews')
kitchen = open_unlabeled_amazon("datasets/amazon-multi-domain/kitchen_&_housewares/")
print('generating kitchen embeddings')
generate_word2vec_embeddings(kitchen,"kitchen", workers=workers)
kitchen = None

print('tokenizing semeval_2013 tweets')
semeval_2013 = open_twitter("datasets/semeval_2013/")
print('generating semeval_2013 embeddings')
generate_word2vec_embeddings(semeval_2013,"semeval_2013", workers=workers)
semeval_2013 = None

print('tokenizing semeval_2016 tweets')
semeval_2016 = open_twitter("datasets/semeval_2016/")
print('generating semeval_2016 embeddings')
generate_word2vec_embeddings(semeval_2016,"semeval_2016", workers=workers)
