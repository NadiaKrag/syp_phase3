import sys, os, re
import numpy as np
from itertools import combinations, product
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import Utils.Datasets as dsets
from Utils.WordVecs import WordVecs
from Utils.Representations import words
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def print_gold(gold_y, outfile):
    with open(outfile, 'w') as out:
        for l in gold_y:
            out.write('{0}\n'.format(l))


def print_info(src, trgs, alpha, batch_size):
    print('{0} --> {1}'.format(src, trgs))
    print('batch: {0}'.format(batch_size))
    print('alpha: {0}'.format(alpha))


class MLSE_domain(nn.Module):
    """MLSE PyTorch Model Class.

    Note
    ----
    Custom modules are subclasses of torch.nn.Module

    Parameters
    ----------
    domain_vecs : WordVecs or GloveVecs
        Source and target embeddings.
    src_name : string
        Name of source domain.
    output_dim : int, optional
        Output dimension.

    Attributes
    ----------
    history : dict
        Contains loss, dev_cosine, dev_f1, cross_f1 from each epoch.
    """
    def __init__(self, domain_vecs, src_name, output_dim=2):
        super().__init__()

        ## Embeddings
        self.src_name = src_name
        self.embs = dict()
        self.dw2idx = dict()
        self.didx2w = dict()
        for domain, vecs in domain_vecs.items():
            # Domain embedding layer
            self.embs[domain] = nn.Embedding(vecs.vocab_length, vecs.vector_size)
            # Set weights to pretrained domain embeddings
            self.embs[domain].weight.data.copy_(torch.from_numpy(vecs._matrix))
            # Dont update embeddings
            self.embs[domain].weight.requires_grad = False
            # Dict map of word -> index
            self.dw2idx[domain] = vecs._w2idx
            # Dict map of index -> word
            self.didx2w[domain] = vecs._idx2w
        self.embs = nn.ModuleDict(self.embs)

        ## Projections
        self.ms = dict()
        for domain, vecs in domain_vecs.items():
            # Linear layer for projection (same as a Dense layer in keras); the weights are M, e.g. z_i = S_si @ M
            self.ms[domain] = nn.Linear(vecs.vector_size, vecs.vector_size, bias=False)
        self.ms = nn.ModuleDict(self.ms)

        ## Classification
        # Linear layer for relu function (to introduce some non-linearity?)
        self.clf2 = nn.Linear(domain_vecs[self.src_name].vector_size, domain_vecs[self.src_name].vector_size)
        # Linear layer for classification; weights are P + bias
        self.clf = nn.Linear(domain_vecs[self.src_name].vector_size, output_dim)

        ## Losses
        # Sentiment loss
        self.ce_loss = nn.CrossEntropyLoss()
        # Projection loss
        self.mse_loss = nn.MSELoss()

        ## Optimizer
        # self.parameters adds parameters of each layer that should be optimized
        self.optim = torch.optim.Adam(self.parameters())

        ## History
        # Save stuff from epochs
        self.history = {'train_loss':  [],
                        'train_prj_loss': [],
                        'train_clf_loss': [],
                        'dev_loss': [],
                        'dev_prj_loss': [],
                        'dev_clf_loss': [],
                        'src_dev_f1': [],
                        'dev_cosine': [],
                        'dev_cm': []}

    def dump_weights(self, outfile):
        torch.save((OrderedDict((key, val) for key, val in self.state_dict().items() if "embs" not in key)), outfile)

    def load_weights(self, weight_file):
        self.load_state_dict(torch.load(weight_file), strict=False)

    def project(self, domain_tokens, domain):
        """Project tokens into shared space.

        Parameters
        ----------
        domain_tokens : array_like
            Array of tokens.
        domain : string
            Name of the domain.

        Returns
        ----------
        proj : 2d array
            Projected vectors.
        """
        # Map list of tokens to emb indexes
        domain_idx = self.tokens_to_idx(domain_tokens, self.dw2idx[domain])
        domain_idx = Variable(domain_idx).to(device)
        # Get embeddings
        domain_embed = self.embs[domain](domain_idx)
        # Use projection layer to project to shared space
        return self.ms[domain](domain_embed)

    def projection_loss(self, lexicon_tokens, weighted=False, beta=0.0):
        proj_loss = 0
        losses = list()
        max_mse = 0
        for domain in lexicon_tokens:
            src_proj = self.project(lexicon_tokens[domain][0], self.src_name)
            trg_proj = self.project(lexicon_tokens[domain][1], domain)
            # Mean Squared Error
            domain_loss = self.mse_loss(src_proj, trg_proj)
            proj_loss += domain_loss
            if weighted:
                losses.append(domain_loss)
            elif domain_loss > max_mse:
                max_mse = domain_loss
        if weighted:
            # Each domain is weighted by their contribution to total mse
            return sum(loss**2/proj_loss for loss in losses)
        else:
            # Average mse per domain
            proj_loss /= len(lexicon_tokens)
            # Beta weighs average mse vs. greatest mse
            return (1-beta)*proj_loss + beta*max_mse

    def tokens_to_idx(self, tokens, vocab):
        """
        Maps single token list to embedding index list.
        Uses index 0 if the token is not in embedding vocab (OOV vector)
        """
        sent = [vocab.get(t, 0) for t in tokens]
        return torch.LongTensor(np.array(sent))

    def all_tokens_to_idx(self, X, vocab):
        """
        Map each token list in X to an embedding index list
        """
        return [self.tokens_to_idx(x, vocab) for x in X]

    def ave_vecs(self, X, domain):
        """
        Averages embeddings in each text, i.e. compute a_i vectors.
        """
        vecs = []
        idxs = self.all_tokens_to_idx(X, self.dw2idx[domain])
        for idx in idxs:
            vecs.append(self.embs[domain](Variable(idx).to(device)).mean(0))
        return torch.stack(vecs)

    def predict(self, X, domain):
        """Predicts sentiment with domain-specific projected embedding vectors.

        Parameters
        ----------
        X : array_like
            Array of tokens.
        domain : string
            Name of the domain.

        Returns
        ----------
        What does it return?
        """
        # Average embeddings for each sentence (compute a_i vectors)
        X_ave = self.ave_vecs(X, domain)
        # project to joint space (compute z_i vectors): z_i = a_i @ M
        X_proj = self.ms[domain](X_ave)
        # Extra relu layer
        X_proj = F.relu(self.clf2(X_proj))
        # Classify y_i = softmax(z_i @ P + bias)
        out = F.softmax(self.clf(X_proj), dim=1)
        # 2 dim output of sentiment label prob
        return out

    def classification_loss(self, X, y, domain):
        pred = self.predict(X, domain)
        y = Variable(torch.from_numpy(y)).to(device)
        # Cross-entropy loss
        loss = self.ce_loss(pred, y)
        return loss

    def full_loss(self, lexicon_tokens, X, y,
                  alpha=.5):
        """
        This is the combined projection and classification loss
        alpha controls the amount of weight given to each
        loss term.
        """

        prj_loss = self.projection_loss(lexicon_tokens)
        clf_loss = self.classification_loss(X, y, self.src_name)
        return self.combine_losses(prj_loss, clf_loss, alpha)

    def combine_losses(self, prj_loss, clf_loss, alpha):
        return alpha * prj_loss + (1 - alpha) * clf_loss

    def fit(self, lexicons,
            src_data,
            trg_datasets,
            weight_dir='models',
            batch_size=40,
            epochs=200,
            alpha=0.5):

        X, y = src_data._Xtrain, src_data._ytrain
        lexicon_train = {domain: (lex._Xtrain, lex._ytrain) for domain, lex in lexicons.items()}
        lexicon_dev = {domain: (lex._Xdev, lex._ydev) for domain, lex in lexicons.items()}
        num_batches = np.ceil(len(X) / batch_size)
        src_best_f1 = 0
        for epoch in range(epochs):
            total_batch_prj_loss = 0
            total_batch_clf_loss = 0
            total_batch_loss = 0
            for idx in range(0, len(X), batch_size):
                batch_X = X[idx:idx+batch_size]
                batch_y = y[idx:idx+batch_size]
                # Clear (old) gradients
                self.optim.zero_grad()
                # Compute losses
                prj_loss = self.projection_loss(lexicon_train)
                clf_loss = self.classification_loss(batch_X, batch_y, self.src_name)
                loss = self.combine_losses(prj_loss, clf_loss, alpha)
                # Compute gradient
                loss.backward()
                # Update parameters
                self.optim.step()

                total_batch_prj_loss += prj_loss.item()
                total_batch_clf_loss += clf_loss.item()
                total_batch_loss += loss.item()

            self.compute_dev_metrics(lexicon_dev, src_data, trg_datasets)
            self.history["train_loss"].append(total_batch_loss / num_batches)
            self.history["train_prj_loss"].append(total_batch_prj_loss / num_batches)
            self.history["train_clf_loss"].append(total_batch_clf_loss / num_batches)

            src_dev_f1 = self.history["src_dev_f1"][-1]
            if src_dev_f1 > src_best_f1:
                src_best_f1 = src_dev_f1
                weight_file = os.path.join(weight_dir, '{0}epochs-{1}batchsize-{2}alpha-{3:.3f}f1.pt'.format(epoch, batch_size, alpha, src_best_f1))
                self.dump_weights(weight_file)

            avg_dev_cosine = np.nanmean(list(self.history["dev_cosine"][-1].values()))
            sys.stdout.write('\repoch {0} full_loss: {1:.3f}  avg_dev_proj_cosine: {2:.3f}  src_dev_f1: {3:.3f}'.format(
                epoch, total_batch_loss/num_batches, avg_dev_cosine, src_dev_f1))
            sys.stdout.flush()
        history_fname = os.path.join(weight_dir, '{1}batchsize-{2}alpha-{3:.3f}f1.txt'.format(epochs, batch_size, alpha, src_best_f1))
        self.dump_history(history_fname)
        print()

    def compute_dev_metrics(self, lexicon_dev, src_data, trg_datasets, labels=[0, 1]):
        cms = dict()
        cos = dict()
        # check source dev f1
        xp = self.predict(src_data._Xdev, self.src_name).cpu().data.numpy().argmax(1)
        src_dev_f1 = f1_score(src_data._ydev, xp, average='macro')

        dev_prj_loss = self.projection_loss(lexicon_dev)
        dev_clf_loss = self.classification_loss(src_data._Xdev, src_data._ydev, self.src_name)
        dev_loss = self.combine_losses(dev_prj_loss, dev_clf_loss, alpha).item()
        dev_prj_loss = dev_prj_loss.item()
        dev_clf_loss = dev_clf_loss.item()

        src_dev_cm = confusion_matrix(src_data._ydev, xp, labels).ravel()

        cms[self.src_name] = src_dev_cm
        cos[self.src_name] = 1

        for domain in lexicon_dev:
            src_proj = self.project(lexicon_dev[domain][0], self.src_name)
            trg_proj = self.project(lexicon_dev[domain][1], domain)
            cos[domain] = self.cos(src_proj, trg_proj).item()

            pred = self.predict(trg_datasets[domain]._Xdev, domain).cpu().data.numpy().argmax(1)

            trg_dev_cm = confusion_matrix(trg_datasets[domain]._ydev, pred, labels).ravel()
            cms[domain] = trg_dev_cm

        self.history['dev_loss'].append(dev_loss)
        self.history['dev_prj_loss'].append(dev_prj_loss)
        self.history['dev_clf_loss'].append(dev_clf_loss)
        self.history['src_dev_f1'].append(src_dev_f1)
        self.history['dev_cosine'].append(cos)
        self.history['dev_cm'].append(cms)

    def dump_history(self, history_fname):
        with open(history_fname, 'w') as f:
            f.write(','.join([
                            'epoch',
                            'train_loss',
                            'train_prj_loss',
                            'train_clf_loss',
                            'dev_loss',
                            'dev_prj_loss',
                            'dev_clf_loss',
                            'domain',
                            'tn',
                            'fp',
                            'fn',
                            'tp',
                            'dev_cosine'
                            ]) + '\n')
            for i in range(len(self.history["src_dev_f1"])):
                f.write("{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{},{},{},{},{},{:.4f}\n".format(
                    i,
                    self.history['train_loss'][i],
                    self.history['train_prj_loss'][i],
                    self.history['train_clf_loss'][i],
                    self.history['dev_loss'][i],
                    self.history['dev_prj_loss'][i],
                    self.history['dev_clf_loss'][i],
                    self.src_name,
                    *self.history['dev_cm'][i][self.src_name],
                    self.history['dev_cosine'][i][self.src_name],
                    ))
                for j, domain in enumerate(self.embs.keys()):
                    if domain == self.src_name:
                        continue
                    f.write("{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{},{},{},{},{},{:.4f}\n".format(
                        i,
                        self.history['train_loss'][i],
                        self.history['train_prj_loss'][i],
                        self.history['train_clf_loss'][i],
                        self.history['dev_loss'][i],
                        self.history['dev_prj_loss'][i],
                        self.history['dev_clf_loss'][i],
                        domain,
                        *self.history['dev_cm'][i][domain],
                        self.history['dev_cosine'][i][domain],
                        ))

    def cos(self, x, y):
        c = nn.CosineSimilarity()
        return c(x, y).mean()

    # def get_most_probable_translations(self, src_word, n=5):
    #     px = self.m(self.semb.weight)
    #     py = self.mp(self.temb.weight)
    #     # Matrix multipl. of src word projected vector with target projected embeddings
    #     preds = torch.mm(py, (px[self.sw2idx[src_word]]).unsqueeze(1))
    #     # Remove a dim
    #     preds = preds.squeeze(1)
    #     preds = preds.data.numpy()
    #     # return the n words with largest dot products ~ most similar words
    #     return [self.tidx2w[i] for i in preds.argsort()[-n:]]

    def confusion_matrix(self, X, y, domain, labels=[0, 1]):
        pred = self.predict(X, domain).data.numpy().argmax(1)
        cm = confusion_matrix(y, pred, labels)
        return cm

    def evaluate(self, X, y, domain, outfile=None):
        pred = self.predict(X, domain).cpu().data.numpy().argmax(1)
        acc = accuracy_score(y, pred)
        f1 = f1_score(y, pred, average='macro')
        if outfile:
            with open(outfile, 'w') as out:
                for i in pred:
                    out.write('{0}\n'.format(i))
        return acc, f1


def get_best_run(weightdir, batch_size=None, alpha=None):
    """
    This returns the best dev f1, parameters, and weights from the models
    found in the weightdir.
    """
    best_params = []
    best_f1 = 0.0
    best_weights = ''
    for fname in os.listdir(weightdir):
        if ".pt" not in fname:
            continue
        epochs = int(re.findall('[0-9]+', fname.split('-')[-4])[0])
        batch = int(re.findall('[0-9]+', fname.split('-')[-3])[0])
        alp = float(re.findall('0.[0-9]+', fname.split('-')[-2])[0])
        f1 = float(re.findall('0.[0-9]+', fname.split('-')[-1])[0])
        if batch_size and alpha:
            if batch == batch_size and alp == alpha:
                if f1 > best_f1:
                    best_params = [epochs, batch, alp]
                    best_f1 = f1
                    weights = os.path.join(weightdir, fname)
                    best_weights = weights
        elif batch_size:
            if batch == batch_size:
                if f1 > best_f1:
                    best_params = [epochs, batch, alp]
                    best_f1 = f1
                    weights = os.path.join(weightdir, fname)
                    best_weights = weights
        elif alpha:
            if alp == alpha:
                if f1 > best_f1:
                    best_params = [epochs, batch, alp]
                    best_f1 = f1
                    weights = os.path.join(weightdir, fname)
                    best_weights = weights
        else:
            if f1 > best_f1:
                best_params = [epochs, batch, alp]
                best_f1 = f1
                weights = os.path.join(weightdir, fname)
                best_weights = weights

    return best_f1, best_params, best_weights


if __name__ == '__main__':
    # Num classes to predict
    outdim = 2
    # Training epochs
    epochs = 200
    # Translation pairs
    trans = 'lexicons/general_vocab.txt'
    # Where to save predictions and model weights
    base_dir = 'results/two_targets_1'
    base_weights_dir = os.path.join(base_dir, 'weights')
    base_predictions_dir = os.path.join(base_dir, 'predictions')

    print('Processing data...')
    books = dsets.BookDataset(None, rep=words, one_hot=False, binary=True)
    dvd = dsets.DVDDataset(None, rep=words, one_hot=False, binary=True)
    electronics = dsets.ElectronicsDataset(None, rep=words, binary=True, one_hot=False)
    kitchen = dsets.KitchenDataset(None, rep=words, binary=True, one_hot=False)

    semeval2013 = dsets.SemevalDataset('datasets/semeval_2013', None,
                                       binary=True, rep=words,
                                       one_hot=False)

    semeval2016 = dsets.SemevalDataset('datasets/semeval_2016', None,
                                       binary=True, rep=words,
                                       one_hot=False)

    data = {'books': (books, 'embeddings/amazon-sg-300.txt'),
            'dvd': (dvd, 'embeddings/amazon-sg-300.txt'),
            'electronics': (electronics, 'embeddings/amazon-sg-300.txt'),
            'kitchen': (kitchen, 'embeddings/amazon-sg-300.txt'),
            'semeval2013': (semeval2013, 'embeddings/twitter_embeddings.txt'),
            'semeval2016': (semeval2016, 'embeddings/twitter_embeddings.txt')
            }

    domains = sorted(data)
    ## Normal combinations
    # trg_combs = []
    # for i in range(1, len(domains)):
    #     trg_combs += list(combinations(domains, i))

    ##
    trg_combs = list(product(['books', 'dvd', 'electronics', 'kitchen'], ['semeval2013', 'semeval2016']))
    print(len(trg_combs))

    # iterate over all datasets as train and test
    for src_name, (src_data, src_emb_file) in data.items():
        if src_name in ["semeval2013", "semeval2016"]:
            continue

        # Source embeddings
        print('Loading source embeddings...')
        src_emb = WordVecs(src_emb_file)
        for comb in trg_combs:
            if src_name in comb:
                continue
            domain_vecs = {src_name: src_emb}

            trg_names = []
            trg_datasets = dict()
            lexicons = dict()
            print('Loading target embeddings and lexicon...')
            for trg_name in comb:
                _, trg_emb_file = data[trg_name]
                domain_vecs[trg_name] = WordVecs(trg_emb_file)
                trg_names.append(trg_name)
                preddir = os.path.join(base_predictions_dir, trg_name)
                os.makedirs(preddir, exist_ok=True)
                trg_datasets[trg_name] = data[trg_name][0]

                lexicons[trg_name] = dsets.LexiconDataset(trans, src_emb, domain_vecs[trg_name])

            trg_str = '_'.join(trg_names)

            savedir = os.path.join(base_weights_dir, src_name, trg_str)
            # # Create folders
            os.makedirs(savedir, exist_ok=True)
            for alpha in [0.1, 0.3, 0.5, 0.7, 0.9]:
                for batch_size in [100, 200, 500]:

                    print_info(src_name, trg_names, alpha, batch_size)

                    clf = MLSE_domain(domain_vecs, src_name)
                    clf = clf.to(device)

                    clf.fit(lexicons,
                            src_data,
                            trg_datasets,
                            weight_dir=savedir,
                            batch_size=batch_size,
                            epochs=epochs,
                            alpha=alpha)

                    best_f1, best_params, best_weights = get_best_run(savedir, alpha=alpha, batch_size=batch_size)
                    clf.load_weights(best_weights)
                    #
                    for trg_name in trg_names:
                        preddir = os.path.join(base_predictions_dir, trg_name)
                        trg_data = data[trg_name][0]

                        outfile = os.path.join(base_predictions_dir,
                                               trg_name,
                                               '{0}-{1}-epochs{2}-batch{3}-alpha{4}_srcf1{5}.txt'.format(
                                                   src_name,
                                                   trg_str,
                                                   best_params[0],
                                                   best_params[1],
                                                   best_params[2],
                                                   best_f1
                                                   )
                                               )
                        acc, f1 = clf.evaluate(trg_data._Xtest, trg_data._ytest, trg_name, outfile)

                        print("{}: test acc: {:.3f}, test f1: {:.3f}".format(trg_name, acc, f1))

                        # print both gold
                        print_gold(trg_data._ytest, os.path.join(base_predictions_dir, '{0}.gold.txt'.format(trg_name)))
                    print('--------------')
