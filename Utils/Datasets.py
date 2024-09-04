import os, re
import numpy as np
from Utils.Representations import words, ave_vecs


def train_dev_test_split(x, train=.7, dev=.1):
    train_idx = int(len(x) * train)
    dev_idx = int(len(x) * (train + dev))
    return x[:train_idx], x[train_idx:dev_idx], x[dev_idx:]

# class LexiconDataset:
#     def __init__(self, translation_dictionary, embs):
#         self._train, self._dev = self.open_data(translation_dictionary, embs)
#
#     def open_data(self, translation_dictionary, embs):
#         with open(translation_dictionary) as f:
#             emb_domains = embs.keys()
#             lex_domains = f.readline().split()
#             if not set(emb_domains).issubset(set(lex_domains)):
#                 raise ValueError("Lexicon does not contain all domains in embeddings.")
#             lex_domains = {domain: i for i, domain in enumerate(lex_domains)}
#             word_lists = {dom:[] for dom in emb_domains}
#             for line in f:
#                 words = line.split()
#                 try:
#                     for domain in emb_domains:
#                         col_i = lex_domains[domain]
#                         _ = embs[domain][words[col_i]]
#                     for domain in emb_domains:
#                         word_lists[domain].append(words[col_i])
#                 except KeyError:
#                     pass
#
#         train = dict()
#         dev = dict()
#         for domain, words in word_lists.items():
#             xtr, xdev, _ = train_dev_test_split(words, 0.9, 0.1)
#             train[domain] = xtr
#             dev[domain] = xdev
#         return train, dev

class LexiconDataset():
    def __init__(self, translation_dictionary, src_vecs, trg_vecs):
        (self._Xtrain, self._Xdev, self._ytrain,
         self._ydev) = self.getdata(translation_dictionary, src_vecs, trg_vecs)

    def getdata(self, translation_dictionary, src_vecs, trg_vecs):
        x, y = [], []
        with open(translation_dictionary) as f:
            for line in f:
                try:
                    src, trg = line.split()
                    _ = src_vecs[src]
                    _ = trg_vecs[trg]
                    x.append(src)
                    y.append(trg)
                except:
                    pass
        xtr, xdev, _ = train_dev_test_split(x, 0.9, 0.1)
        ytr, ydev, _ = train_dev_test_split(y, 0.9, 0.1)
        return xtr, xdev, ytr, ydev


class GeneralDataset(object):
    """This class takes as input the directory of a corpus annotated for 4 levels
    sentiment. This directory should have 4 .txt files: strneg.txt, neg.txt,
    pos.txt and strpos.txt. It also requires a word embedding model, such as
    those used in word2vec or GloVe.

    binary: instead of 4 classes you have binary (pos/neg). Default is False

    one_hot: the y labels are one hot vectors where the correct class is 1 and
             all others are 0. Default is True.

    dtype: the dtype of the np.array for each vector. Default is np.float32.

    rep: this determines how the word vectors are represented.

         sum_vecs: each sentence is represented by one vector, which is
                    the sum of each of the word vectors in the sentence.

         ave_vecs: each sentence is represented as the average of all of the
                    word vectors in the sentence.

         idx_vecs: each sentence is respresented as a list of word ids given by
                    the word-2-idx dictionary.
    """
    def __init__(self, DIR, model, binary=False, one_hot=True,
                 dtype=np.float32, rep=ave_vecs, lowercase=True):

        self.rep = rep
        self.one_hot = one_hot
        self.lowercase = lowercase

        Xtrain, Xdev, Xtest, ytrain, ydev, ytest = self.open_data(DIR, model, binary, rep)


        self._Xtrain = Xtrain
        self._ytrain = ytrain
        self._Xdev = Xdev
        self._ydev = ydev
        self._Xtest = Xtest
        self._ytest = ytest
        self._num_examples = len(self._Xtrain)

    def to_array(self, y, N):
        '''
        converts an integer-based class into a one-hot array
        y = the class integer
        N = the number of classes
        '''
        return np.eye(N)[y]

    def open_data(self, DIR, model, binary, rep):
        if binary:
            ##################
            # Binary         #
            ##################
            train_neg = getMyData(os.path.join(DIR, 'train/neg.txt'),
                                  0, model, encoding='latin',
                                  representation=rep)
            train_neg2 = getMyData(os.path.join(DIR, 'train/strneg.txt'),
                                  0, model, encoding='latin',
                                  representation=rep)
            train_pos = getMyData(os.path.join(DIR, 'train/pos.txt'),
                                  1, model, encoding='latin',
                                  representation=rep)
            train_pos2 = getMyData(os.path.join(DIR, 'train/strpos.txt'),
                                  1, model, encoding='latin',
                                  representation=rep)
            dev_neg = getMyData(os.path.join(DIR, 'dev/neg.txt'),
                                0, model, encoding='latin',
                                representation=rep)
            dev_neg2 = getMyData(os.path.join(DIR, 'dev/strneg.txt'),
                                0, model, encoding='latin',
                                representation=rep)
            dev_pos = getMyData(os.path.join(DIR, 'dev/pos.txt'),
                                1, model, encoding='latin',
                                representation=rep)
            dev_pos2 = getMyData(os.path.join(DIR, 'dev/strpos.txt'),
                                1, model, encoding='latin',
                                representation=rep)
            test_neg = getMyData(os.path.join(DIR, 'test/neg.txt'),
                                 0, model, encoding='latin',
                                 representation=rep)
            test_neg2 = getMyData(os.path.join(DIR, 'test/strneg.txt'),
                                 0, model, encoding='latin',
                                 representation=rep)
            test_pos = getMyData(os.path.join(DIR, 'test/pos.txt'),
                                 1, model, encoding='latin',
                                 representation=rep)
            test_pos2 = getMyData(os.path.join(DIR, 'test/strpos.txt'),
                                 1, model, encoding='latin',
                                 representation=rep)


            traindata = train_pos + train_pos2 + train_neg + train_neg2
            devdata = dev_pos + dev_pos2 + dev_neg + dev_neg2
            testdata = test_pos + test_pos2 + test_neg + test_neg2

            # Set up vocab now
            self.vocab = set()

            # Training data
            Xtrain = [data for data, y in traindata]
            if self.lowercase:
                Xtrain = [[w.lower() for w in sent] for sent in Xtrain]
            if self.one_hot is True:
                ytrain = [self.to_array(y, 2) for data, y in traindata]
            else:
                ytrain = [y for data, y in traindata]
            self.vocab.update(set([w for i in Xtrain for w in i]))

            # Dev data
            Xdev = [data for data, y in devdata]
            if self.lowercase:
                Xdev = [[w.lower() for w in sent] for sent in Xdev]
            if self.one_hot is True:
                ydev = [self.to_array(y, 2) for data, y in devdata]
            else:
                ydev = [y for data, y in devdata]
            self.vocab.update(set([w for i in Xdev for w in i]))

            # Test data
            Xtest = [data for data, y in testdata]
            if self.lowercase:
                Xtest = [[w.lower() for w in sent] for sent in Xtest]
            if self.one_hot is True:
                ytest = [self.to_array(y, 2) for data, y in testdata]
            else:
                ytest = [y for data, y in testdata]
            self.vocab.update(set([w for i in Xtest for w in i]))
        else:
            ##################
            # 4 CLASS        #
            ##################
            train_strneg = getMyData(os.path.join(DIR, 'train/strneg.txt'),
                                  0, model, encoding='latin',
                                  representation=rep)
            train_strpos = getMyData(os.path.join(DIR, 'train/strpos.txt'),
                                  3, model, encoding='latin',
                                  representation=rep)
            train_neg = getMyData(os.path.join(DIR, 'train/neg.txt'),
                                  1, model, encoding='latin',
                                  representation=rep)
            train_pos = getMyData(os.path.join(DIR, 'train/pos.txt'),
                                  2, model, encoding='latin',
                                  representation=rep)
            dev_strneg = getMyData(os.path.join(DIR, 'dev/strneg.txt'),
                                0, model, encoding='latin',
                                representation=rep)
            dev_strpos = getMyData(os.path.join(DIR, 'dev/strpos.txt'),
                                3, model, encoding='latin',
                                representation=rep)
            dev_neg = getMyData(os.path.join(DIR, 'dev/neg.txt'),
                                1, model, encoding='latin',
                                representation=rep)
            dev_pos = getMyData(os.path.join(DIR, 'dev/pos.txt'),
                                2, model, encoding='latin',
                                representation=rep)
            test_strneg = getMyData(os.path.join(DIR, 'test/strneg.txt'),
                                 0, model, encoding='latin',
                                 representation=rep)
            test_strpos = getMyData(os.path.join(DIR, 'test/strpos.txt'),
                                 3, model, encoding='latin',
                                 representation=rep)
            test_neg = getMyData(os.path.join(DIR, 'test/neg.txt'),
                                 1, model, encoding='latin',
                                 representation=rep)
            test_pos = getMyData(os.path.join(DIR, 'test/pos.txt'),
                                 2, model, encoding='latin',
                                 representation=rep)

            traindata = train_pos + train_neg + train_strneg + train_strpos
            devdata = dev_pos + dev_neg + dev_strneg + dev_strpos
            testdata = test_pos + test_neg + test_strneg + test_strpos

            self.vocab = set()

            # Training data
            Xtrain = [data for data, y in traindata]
            if self.lowercase:
                Xtrain = [[w.lower() for w in sent] for sent in Xtrain]
            if self.one_hot is True:
                ytrain = [self.to_array(y, 4) for data, y in traindata]
            else:
                ytrain = [y for data, y in traindata]
            self.vocab.update(set([w for i in Xtrain for w in i]))

            # Dev data
            Xdev = [data for data, y in devdata]
            if self.lowercase:
                Xdev = [[w.lower() for w in sent] for sent in Xdev]
            if self.one_hot is True:
                ydev = [self.to_array(y, 4) for data, y in devdata]
            else:
                ydev = [y for data, y in devdata]
            self.vocab.update(set([w for i in Xdev for w in i]))

            # Test data
            Xtest = [data for data, y in testdata]
            if self.lowercase:
                Xtest = [[w.lower() for w in sent] for sent in Xtest]
            if self.one_hot is True:
                ytest = [self.to_array(y, 4) for data, y in testdata]
            else:
                ytest = [y for data, y in testdata]
            self.vocab.update(set([w for i in Xtest for w in i]))

        if self.rep is not words:
            Xtrain = np.array(Xtrain)
            Xdev = np.array(Xdev)
            Xtest = np.array(Xtest)
        ytrain = np.array(ytrain)
        ydev = np.array(ydev)
        ytest = np.array(ytest)

        return Xtrain, Xdev, Xtest, ytrain, ydev, ytest

##########################################################################

class AmazonDataset(GeneralDataset):

    def open_data(self, DIR, model, binary, rep):
        neg = open(os.path.join(DIR,'negative.review')).read()
        pos = open(os.path.join(DIR,'positive.review')).read()

        pos = pos.split('<review>')[1:]
        neg = neg.split('<review>')[1:]

        posX = [self.get_between(l, '<review_text>\n', '\n</review_text>') for l in pos]
        negX = [self.get_between(l, '<review_text>\n', '\n</review_text>') for l in neg]

        posX = [self.clean_str(s) for s in posX]
        negX = [self.clean_str(s) for s in negX]

        if binary:
            posy = [1] * len(posX)
            negy = [0] * len(negX)
            if self.one_hot is True:
                posy = [self.to_array(y, 2) for y in posy]
                negy = [self.to_array(y, 2) for y in negy]
        else:
            posy = [float(self.get_between(l, '<rating>\n', '\n</rating>')) for l in pos]
            negy = [float(self.get_between(l, '<rating>\n', '\n</rating>')) for l in neg]
            posy = [self.change_y(y) for y in posy]
            negy = [self.change_y(y) for y in negy]
            if self.one_hot is True:
                posy = [self.to_array(y, 4) for y in posy]
                negy = [self.to_array(y, 4) for y in negy]

        pos = list(zip(posy, posX))
        neg = list(zip(negy, negX))

        train_idx = int(len(pos) * .75)
        dev_idx = int(len(pos) * .8)

        train_neg = neg[:train_idx]
        dev_neg = neg[train_idx:dev_idx]
        test_neg = neg[dev_idx:]

        train_pos = pos[:train_idx]
        dev_pos = pos[train_idx:dev_idx]
        test_pos = pos[dev_idx:]

        train_data = train_pos + train_neg
        dev_data = dev_pos + dev_neg
        test_data = test_pos + test_neg

        ytrain, Xtrain = zip(*train_data)
        Xtrain = [rep(sent, model) for sent in Xtrain]

        ydev, Xdev = zip(*dev_data)
        Xdev = [rep(sent, model) for sent in Xdev]

        ytest, Xtest = zip(*test_data)
        Xtest = [rep(sent, model) for sent in Xtest]


        if self.rep is not words:
            Xtrain = np.array(Xtrain)
            Xdev = np.array(Xdev)
            Xtest = np.array(Xtest)

        ytrain = np.array(ytrain)
        ydev = np.array(ydev)
        ytest = np.array(ytest)

        return Xtrain, Xdev, Xtest, ytrain, ydev, ytest

    def get_between(self, x, l, r):
        mid = x.split(l)[1]
        return mid.split(r)[0]

    def clean_str(self, string, TREC=False):
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

    def change_y(self, y):
        if y == 1.0:
            return 0
        elif y == 2.0:
            return 1
        elif y == 4.0:
            return 2
        elif y == 5.0:
            return 3


###################################################

class BookDataset(AmazonDataset):
    def __init__(self, model, binary=False, one_hot=True,
             dtype=np.float32, rep=ave_vecs):
        DIR = 'datasets/amazon-multi-domain/books'
        super(AmazonDataset, self).__init__(DIR, model, binary, one_hot, dtype, rep)

class DVDDataset(AmazonDataset):
    def __init__(self, model, binary=False, one_hot=True,
             dtype=np.float32, rep=ave_vecs):
        DIR = 'datasets/amazon-multi-domain/dvd'
        super(AmazonDataset, self).__init__(DIR, model, binary, one_hot, dtype, rep)

class ElectronicsDataset(AmazonDataset):
    def __init__(self, model, binary=False, one_hot=True,
             dtype=np.float32, rep=ave_vecs):
        DIR = 'datasets/amazon-multi-domain/electronics'
        super(AmazonDataset, self).__init__(DIR, model, binary, one_hot, dtype, rep)

class KitchenDataset(AmazonDataset):
    def __init__(self, model, binary=False, one_hot=True,
             dtype=np.float32, rep=ave_vecs):
        DIR = 'datasets/amazon-multi-domain/kitchen_&_housewares'
        super(AmazonDataset, self).__init__(DIR, model, binary, one_hot, dtype, rep)

###################################################

class SemevalDataset(GeneralDataset):

    def convert_ys(self, y, binary):
        if 'negative' in y:
            return 0
        elif 'neutral' in y:
            return 1
        elif 'objective' in y:
            return 1
        elif 'positive' in y:
            if binary:
                return 1
            else:
                return 2

    def open_data(self, DIR, model, binary, rep):
        train = []
        for line in open(os.path.join(DIR, 'train.tsv')):
            try:
                idx, sidx, label, tweet = line.split('\t')
            except ValueError:
                idx, label, tweet = line.split('\t', 2)
            if binary:
                if 'neutral' in label or 'objective' in label:
                    pass
                else:
                    train.append((label, tweet))
            else:
                train.append((label, tweet))

        dev = []
        for line in open(os.path.join(DIR, 'dev.tsv')):
            try:
                idx, sidx, label, tweet = line.split('\t')
            except ValueError:
                idx, label, tweet = line.split('\t', 2)
            if binary:
                if 'neutral' in label or 'objective' in label:
                    pass
                else:
                    dev.append((label, tweet))
            else:
                dev.append((label, tweet))

        test = []
        for line in open(os.path.join(DIR, 'test.tsv')):
            try:
                idx, sidx, label, tweet = line.split('\t')
            except ValueError:
                idx, label, tweet = line.split('\t', 2)
            if binary:
                if 'neutral' in label or 'objective' in label:
                    pass
                else:
                    test.append((label, tweet))
            else:
                test.append((label, tweet))

        ytrain, Xtrain = zip(*train)
        ydev,   Xdev   = zip(*dev)
        ytest,  Xtest  = zip(*test)

        Xtrain = [rep(sent, model) for sent in Xtrain]
        ytrain = [self.convert_ys(y, binary) for y in ytrain]

        Xdev = [rep(sent, model) for sent in Xdev]
        ydev = [self.convert_ys(y, binary) for y in ydev]

        Xtest  = [rep(sent, model) for sent in Xtest]
        ytest = [self.convert_ys(y, binary) for y in ytest]

        if self.one_hot:
            if binary:
                ytrain = [self.to_array(y, 2) for y in ytrain]
                ydev = [self.to_array(y,2) for y in ydev]
                ytest = [self.to_array(y,2) for y in ytest]
            else:
                ytrain = [self.to_array(y, 3) for y in ytrain]
                ydev = [self.to_array(y,3) for y in ydev]
                ytest = [self.to_array(y,3) for y in ytest]

        if self.rep is not words:
            Xtrain = np.array(Xtrain)
            Xdev = np.array(Xdev)
            Xtest = np.array(Xtest)

        ytrain = np.array(ytrain)
        ydev = np.array(ydev)
        ytest = np.array(ytest)

        return Xtrain, Xdev, Xtest, ytrain, ydev, ytest
