# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 15:58:12 2019

@author: zzshen
"""
import sys

sys.path.append('..')
import random
import torch
import conll
from doc import Document
from vocab import PAD_ID, ROOT_ID
from dep_vocab import CharVocab, WordVocab, MultiVocab
from data_utils import get_long_tensor, sort_all

from pretrain import Pretrain

from common.options import parse_args


class DataLoader(object):
    def __init__(self, input_src, batch_size, args, pretrain, vocab=None, evaluation=False):
        self.batch_size = batch_size
        self.args = args
        self.eval = evaluation
        self.shuffled = not self.eval

        # check if input source is a file or a Document object
        if isinstance(input_src, str):
            filename = input_src
            assert filename.endswith('conllu'), "Loaded file must be conllu file."
            self.conll, data = self.load_file(filename, evaluation=self.eval)
        elif isinstance(input_src, Document):
            filename = None
            doc = input_src
            self.conll, data = self.load_doc(doc)

        # handle vocab
        if vocab is None:
            self.vocab = self.init_vocab(data)
        else:
            self.vocab = vocab

        self.pretrain_vocab = pretrain.vocab

        # filter and sample data
        if args.get('sample_train', 1.0) < 1.0 and not self.eval:
            keep = int(args['sample_train'] * len(data))
            data = random.sample(data, keep)
            print("Subsample training set with rate {}".format(args['sample_train']))

        data = self.preprocess(data, self.vocab, self.pretrain_vocab, args)

        if self.shuffled:
            random.shuffle(data)
        self.num_examples = len(data)

        # chunk into batches
        self.data = self.chunk_batches(data)
        if filename is not None:
            print("{} batches created for {}.".format(len(self.data), filename))

    def init_vocab(self, data):
        assert self.eval == False  # for eval vocab must exist
        charvocab = CharVocab(data, self.args['shorthand'])
        wordvocab = WordVocab(data, self.args['shorthand'], cutoff=0, lower=True)
        uposvocab = WordVocab(data, self.args['shorthand'], idx=1)
        deprelvocab = WordVocab(data, self.args['shorthand'], idx=3)
        vocab = MultiVocab({'char': charvocab,
                            'word': wordvocab,
                            'upos': uposvocab,
                            'deprel': deprelvocab})
        return vocab

    def preprocess(self, data, vocab, pretrain_vocab, args):
        processed = []
        for sent in data:
            print(sent)
            processed_sent = [[ROOT_ID] + vocab['word'].map([w[0] for w in sent])]
            processed_sent += [[[ROOT_ID]] + [vocab['char'].map([x for x in w[0]]) for w in sent]]
            processed_sent += [[ROOT_ID] + vocab['upos'].map([w[1] for w in sent])]
            processed_sent += [[ROOT_ID] + pretrain_vocab.map([w[0] for w in sent])]
            processed_sent += [[int(w[2]) for w in sent]]
            processed_sent += [vocab['deprel'].map([w[3] for w in sent])]
            processed.append(processed_sent)
            break
        return processed

    def chunk_batches(self, data):
        res = []

        if not self.eval:
            # sort sentences (roughly) by length for better memory utilization
            data = sorted(data, key=lambda x: len(x[0]), reverse=random.random() > .5)

        current = []
        currentlen = 0
        for x in data:
            if len(x[0]) + currentlen > self.batch_size:
                res.append(current)
                current = []
                currentlen = 0
            current.append(x)
            currentlen += len(x[0])

        if currentlen > 0:
            res.append(current)

        return res

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        """ Get a batch with index. """
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError
        batch = self.data[key]
        batch_size = len(batch)

        batch = list(zip(*batch))

        assert len(batch) == 6

        # sort sentences by lens for easy RNN operations
        lens = [len(x) for x in batch[0]]
        batch, orig_idx = sort_all(batch, lens)

        # sort words by lens for easy char-RNN operations
        batch_words = [w for sent in batch[1] for w in sent]
        word_lens = [len(x) for x in batch_words]

        batch_words, word_orig_idx = sort_all([batch_words], word_lens)
        batch_words = batch_words[0]
        word_lens = [len(x) for x in batch_words]

        # convert to tensors
        words = batch[0]
        words = get_long_tensor(words, batch_size)
        words_mask = torch.eq(words, PAD_ID)

        wordchars = get_long_tensor(batch_words, len(word_lens))
        wordchars_mask = torch.eq(wordchars, PAD_ID)

        upos = get_long_tensor(batch[2], batch_size)
        pretrained = get_long_tensor(batch[3], batch_size)
        sentlens = [len(x) for x in batch[0]]
        head = get_long_tensor(batch[4], batch_size)
        deprel = get_long_tensor(batch[5], batch_size)
        return words, words_mask, wordchars, wordchars_mask, upos, pretrained, head, deprel, orig_idx, word_orig_idx, sentlens, word_lens

    def load_file(self, filename, evaluation=False):
        conll_file = conll.CoNLLFile(filename)
        data = conll_file.get(['word', 'upos', 'head', 'deprel'], as_sentences=True)
        return conll_file, data

    def load_doc(self, doc):
        data = doc.conll_file.get(['word', 'upos', 'head', 'deprel'], as_sentences=True)
        return doc.conll_file, data

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def reshuffle(self):
        data = [y for x in self.data for y in x]
        self.data = self.chunk_batches(data)
        random.shuffle(self.data)


if __name__ == '__main__':
    conll_file_path = '../UD/zh-ud-train.conllu'
    conll_file = conll.CoNLLFile(conll_file_path)
    data = conll_file.get(['word', 'upos', 'head', 'deprel'], as_sentences=True)
    args = parse_args()
    args.train_file = conll_file_path
    args.batch_size = 2
    print(args)
    args = vars(args)
    vec_file = '../Embeds/sdp_vec.pkl'
    pretrain_file = '../save/sdp.pretrain.pt'
    pretrain = Pretrain(pretrain_file, vec_file)
    train_batch = DataLoader(args['train_file'], args['batch_size'], args, pretrain, evaluation=False)

    word = train_batch[1][0]
    words_mask = train_batch[1][1]
    wordchars = train_batch[1][2]
    wordchars_mask = train_batch[1][3]
    pretrained = train_batch[1][5]
    upos = train_batch[1][4]

    print(word)
    print('|pretrained: {}'.format(pretrained))
    print('|word size: {}'.format(word.size()))
    print('|words mask size: {}'.format(words_mask.size()))
    print('|wordchars size: {}'.format(wordchars.size()))
